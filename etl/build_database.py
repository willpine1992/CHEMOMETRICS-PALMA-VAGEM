"""
ETL: Isotermas Vagem.xlsx (baixado ao vivo do Google Drive) -> banco relacional SQLite.

Fonte: Google Drive (File ID 1wT33uLHM6rwS1SpiXJ6c9ypj5SaI2X4c), baixado a cada
execucao via drive_source.fetch_xlsx() -- NAO usa mais a copia estatica local em
"ARTIGO VAGEN/". Isso garante que o banco sempre reflita a versao mais recente
editada pelo usuario no Drive.

Escopo: dois materiais (Palma, Vagem), tratamentos Acido e Base (In Natura e a
aba "tempo ph 6 e 10" foram deixados de fora deste primeiro recorte -- ver
limitacoes no relatorio), quatro tipos de experimento OFAT (pH, massa, tempo,
concentracao+temperatura). Qe e %Remocao sao SEMPRE recalculados aqui pela
formula padrao Qe=(C0-Ce)*V/m , %R=(C0-Ce)/C0*100 (a planilha tem colunas "QE"
proprias, mas elas contem um bug de formula em varias abas -- ex. Tempo PALMA
usa "*0.0244" em vez de "*V" -- entao nunca sao usadas como fonte).

NOTA DE QUALIDADE DE DADOS (corrigida): o dataset de ML anterior
(ml/data/vagem_ml_dataset.csv) usava, por bug, C0=234.657926 (a constante da
aba "pH Vagem") tambem para a aba "Massa Vagem", quando na verdade "Massa
Vagem" tem a sua PROPRIA celula de calibracao (Q6=223.334923). Esse bug foi
corrigido aqui: todas as linhas de "Massa Vagem" (Acido e Base) usam agora
C0_MASSA_VAGEM_CORRETO. Isso muda ligeiramente Qe/%Remocao das 4 linhas
Vagem+Acido de massa que alimentam o pipeline de ML -- espere uma pequena
diferenca em metrics.json/parity/shap em relacao a versao anterior.
"""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive_source import fetch_xlsx  # noqa: E402
import validate  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db", "biosorcao.db")
DB_PATH = os.path.abspath(DB_PATH)
TMP_DB_PATH = DB_PATH + ".tmp"

V_L = 0.02  # volume da solucao (L), constante em todas as abas

SCHEMA = """
CREATE TABLE materials (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL
);

CREATE TABLE treatments (
  id INTEGER PRIMARY KEY,
  material_id INTEGER NOT NULL REFERENCES materials(id),
  name TEXT NOT NULL,
  UNIQUE(material_id, name)
);

CREATE TABLE calibration_curves (
  id INTEGER PRIMARY KEY,
  material_id INTEGER NOT NULL REFERENCES materials(id),
  sheet_source TEXT,
  slope REAL,
  intercept REAL,
  equation_text TEXT,
  notes TEXT
);

CREATE TABLE experiments (
  id INTEGER PRIMARY KEY,
  treatment_id INTEGER NOT NULL REFERENCES treatments(id),
  experiment_type TEXT NOT NULL,
  sheet_source TEXT,
  volume_L REAL,
  notes TEXT
);

CREATE TABLE experiment_runs (
  id INTEGER PRIMARY KEY,
  experiment_id INTEGER NOT NULL REFERENCES experiments(id),
  ph REAL,
  massa_g REAL,
  tempo_min REAL,
  conc_inicial_mgL REAL,
  temperatura_C REAL,
  ce_mgL REAL,
  qe_mgg REAL,
  remocao_pct REAL,
  is_assumed_tempo INTEGER DEFAULT 0,
  is_assumed_temp INTEGER DEFAULT 0,
  is_assumed_ph INTEGER DEFAULT 0,
  obs TEXT
);

CREATE TABLE isotherm_params (
  id INTEGER PRIMARY KEY,
  treatment_id INTEGER NOT NULL REFERENCES treatments(id),
  temperature_C REAL,
  model TEXT,             -- 'Langmuir' (linearizacao Ce/Qe vs Ce, como a planilha calcula),
                           -- 'Langmuir (NLS)' (regressao nao-linear direta Qe=Qmax*KL*Ce/(1+KL*Ce),
                           -- recalculada aqui a partir dos pontos brutos -- ver isotherm_raw_points),
                           -- 'Freundlich','Temkin','DKR','Harkin-Jura','Halsey'
  qmax REAL,              -- populado so para model in ('Langmuir','Langmuir (NLS)')
  kl REAL,                -- idem
  rl REAL,                -- fator de separacao; NULL quando a planilha nao reporta (caso de Vagem)
  r2 REAL,                -- R2 do ajuste, para qualquer modelo
  params_json TEXT,       -- TODOS os parametros do modelo, com os nomes originais da planilha
                           -- (cobre Freundlich/Temkin/DKR/Harkin-Jura/Halsey, que tem parametros
                           -- proprios sem coluna dedicada aqui)
  notes TEXT,             -- alerta de qualidade: ajuste instavel por linearizacao, dados brutos
                           -- nao-monotonicos, etc. (ver build())
  sheet_source TEXT
);

CREATE TABLE isotherm_raw_points (
  id INTEGER PRIMARY KEY,
  treatment_id INTEGER NOT NULL REFERENCES treatments(id),
  temperature_C REAL,
  ci_mgL REAL,             -- concentracao inicial nominal do ponto da isoterma
  ce_mgL REAL,             -- concentracao de equilibrio media medida
  qe_mgg REAL,             -- capacidade de adsorcao media medida
  sheet_source TEXT
);
"""

NEG_CE_TRUNCATED = 0


def qe_remocao(c0, ce, m_g):
    global NEG_CE_TRUNCATED
    ce_c = ce
    if pd.isna(ce_c):
        return None, None, ce_c
    if ce_c < 0:
        NEG_CE_TRUNCATED += 1
        ce_c = 0.0
    qe = (c0 - ce_c) * V_L / m_g
    remocao = (c0 - ce_c) / c0 * 100.0
    return qe, remocao, ce_c


def extract_rows(df, row_idxs, ph_col, mass_cols, ce_col, c0, fixed, obs, conc_col=None, tempo_col=None):
    """
    row_idxs: lista de indices de linha (pandas, 0-based)
    ph_col: indice de coluna do pH (ou None se pH vem de `fixed`)
    mass_cols: tupla de indices de colunas de massa (m1[,m2[,m3]]) - media usada como massa_g
    ce_col: indice da coluna de concentracao de equilibrio media (Ce)
    conc_col: se setado, concentracao inicial nominal e lida desta coluna por linha (conc/temp); caso
      contrario usa o valor fixo `c0`
    tempo_col: se setado, tempo (min) e lido desta coluna por linha (experimentos de tempo); caso
      contrario usa o valor fixo em `fixed['tempo_min']`
    fixed: dict com chaves possiveis 'ph','tempo_min','temperatura_C' para valores fixos do bloco
    """
    out = []
    for r in row_idxs:
        ph = df.iat[r, ph_col] if ph_col is not None else fixed.get("ph")
        tempo_min = df.iat[r, tempo_col] if tempo_col is not None else fixed.get("tempo_min")
        masses = [df.iat[r, c] for c in mass_cols if pd.notna(df.iat[r, c])]
        if not masses:
            continue
        m_g = float(np.mean(masses))
        ce = df.iat[r, ce_col]
        c0_row = df.iat[r, conc_col] if conc_col is not None else c0
        if pd.isna(ce) or pd.isna(c0_row) or pd.isna(m_g) or pd.isna(tempo_min):
            continue
        qe, remocao, ce_clipped = qe_remocao(c0_row, ce, m_g)
        out.append(dict(
            ph=ph, massa_g=m_g,
            tempo_min=float(tempo_min),
            conc_inicial_mgL=c0_row,
            temperatura_C=fixed.get("temperatura_C"),
            ce_mgL=ce_clipped, qe_mgg=qe, remocao_pct=remocao,
            obs=obs,
        ))
    return out


def _num(v):
    """NaN/None -> None; caso contrario float() nativo (para caber no sqlite3 e no json)."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return None
    return float(v)


def extract_isotherm_langmuir_vagem(df, treatments):
    """
    Aba 'Isotermas VAGEM': unica fonte de parametros de isoterma para Vagem (a aba
    'Parametros VAGEM' e um template em branco, sem nenhum valor preenchido). So tem
    o modelo de Langmuir. Estrutura: 3 blocos de temperatura (15/30/45 C) comecando
    nas colunas 1, 25 e 49; dentro de cada bloco, 3 sub-blocos de 4 colunas para
    Base(+0)/Acido(+4)/In natura(+8) = [Qmax, KL, RL, R2]. RL vem sempre vazio no
    original (nao e reportado como parametro agregado). No bloco de 45 C o cabecalho
    da 4a coluna esta rotulado "RL" por erro de digitacao da planilha, mas os valores
    (ex.: 0.9459/0.9886/0.7886) seguem claramente o padrao de R2 dos outros blocos --
    tratado como R2 aqui, documentado nesta nota.
    """
    temp_blocks = [(15.0, 1), (30.0, 25), (45.0, 49)]
    treat_suboffset = {"Base": 0, "Ácido": 4, "In natura": 8}
    rows = []
    for temp_c, block_col in temp_blocks:
        for sheet_treat, sub in treat_suboffset.items():
            treat_name = "Acido" if sheet_treat == "Ácido" else sheet_treat
            col = block_col + sub
            qmax, kl, rl, r2 = (df.iat[1, col], df.iat[1, col + 1], df.iat[1, col + 2], df.iat[1, col + 3])
            qmax, kl, rl, r2 = _num(qmax), _num(kl), _num(rl), _num(r2)
            params = {"qmax": qmax, "kl": kl, "rl": rl, "r2": r2}
            rows.append(dict(
                treatment_id=treatments[("Vagem", treat_name)], temperature_C=temp_c, model="Langmuir",
                qmax=qmax, kl=kl, rl=rl, r2=r2, params_json=json.dumps(params), notes=None,
                sheet_source="Isotermas VAGEM",
            ))
    return rows


def extract_isotherm_vagem_raw_points(df, treatments):
    """
    Mesma aba 'Isotermas VAGEM', mesmos 3 blocos de temperatura (col 1/25/49) e
    mesmos sub-blocos de tratamento (Base+0/Acido+4/In natura+8) do Langmuir acima,
    mas lendo a tabela de pontos experimentais brutos que fica logo abaixo do
    resumo do ajuste (linha de cabecalho 'Ci'/'Ce medio'/'Qe medio'/... na linha 4,
    dados nas linhas 5-8). A coluna 'Ci' (concentracao inicial nominal) fica 1
    coluna a esquerda do inicio de cada bloco de temperatura (block_col - 1) e e
    compartilhada pelos 3 tratamentos daquele bloco.
    """
    temp_blocks = [(15.0, 1), (30.0, 25), (45.0, 49)]
    treat_suboffset = {"Base": 0, "Ácido": 4, "In natura": 8}
    rows = []
    for temp_c, block_col in temp_blocks:
        ci_col = block_col - 1
        for sheet_treat, sub in treat_suboffset.items():
            treat_name = "Acido" if sheet_treat == "Ácido" else sheet_treat
            ce_col = block_col + sub
            qe_col = ce_col + 1
            for r in range(5, 9):
                ci, ce, qe = _num(df.iat[r, ci_col]), _num(df.iat[r, ce_col]), _num(df.iat[r, qe_col])
                if ci is None or ce is None or qe is None:
                    continue
                rows.append(dict(
                    treatment_id=treatments[("Vagem", treat_name)], temperature_C=temp_c,
                    ci_mgL=ci, ce_mgL=ce, qe_mgg=qe, sheet_source="Isotermas VAGEM",
                ))
    return rows


def _langmuir_model(ce, qmax, kl):
    return (qmax * kl * ce) / (1.0 + kl * ce)


def refit_langmuir_nls(raw_points):
    """
    Regressao nao-linear direta (scipy.optimize.curve_fit) de Qe = Qmax*KL*Ce/(1+KL*Ce)
    sobre os pontos brutos (Ce,Qe), agrupados por (treatment_id, temperature_C). E o
    metodo estatisticamente preferido sobre a linearizacao Ce/Qe vs Ce que a planilha
    usa (essa ultima amplifica ruido e pode dar Qmax irreal/R2 baixo mesmo quando os
    dados brutos sao bem-comportados -- foi exatamente o caso encontrado em
    Vagem/Base/15C: R2 do metodo linearizado = 0.0286, R2 do NLS = 0.87).

    Retorna uma lista de linhas de isotherm_params (model='Langmuir (NLS)') com uma
    nota de qualidade por bloco: aponta quando o R2 baixo do ajuste linearizado e so
    artefato do metodo (dados brutos monotonicos -> NLS resolve), ou quando o proprio
    Qe bruto nao e monotonico em Ce (problema real nos dados, nao resolvido por
    nenhum metodo de ajuste).
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for p in raw_points:
        grouped[(p["treatment_id"], p["temperature_C"])].append((p["ce_mgL"], p["qe_mgg"]))

    out = []
    notes_by_key = {}
    for (treatment_id, temp_c), pts in grouped.items():
        pts_sorted = sorted(pts, key=lambda t: t[0])
        ce = np.array([p[0] for p in pts_sorted], dtype=float)
        qe = np.array([p[1] for p in pts_sorted], dtype=float)
        monotonic = bool(np.all(np.diff(qe) >= -1e-6))

        try:
            popt, _ = curve_fit(_langmuir_model, ce, qe, p0=[max(qe) * 1.2, 0.01], maxfev=20000)
            qmax_nls, kl_nls = float(popt[0]), float(popt[1])
            qe_pred = _langmuir_model(ce, qmax_nls, kl_nls)
            ss_res = float(np.sum((qe - qe_pred) ** 2))
            ss_tot = float(np.sum((qe - qe.mean()) ** 2))
            r2_nls = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
        except Exception:
            qmax_nls = kl_nls = r2_nls = None

        if not monotonic:
            note = ("Qe bruto nao e monotonico em Ce neste bloco (ha queda de Qe em concentracoes "
                    "mais altas) -- provavel ruido/erro experimental nos pontos originais. Nenhum "
                    "ajuste de Langmuir (linearizado ou NLS) e fisicamente bem definido aqui; "
                    "tratar Qmax/KL/R2 deste bloco com cautela.")
        elif r2_nls is not None and r2_nls >= 0.7:
            note = ("Dados brutos monotonicos e bem-comportados. Se o R2 do model='Langmuir' "
                    "(linearizacao Ce/Qe vs Ce) parecer baixo, e artefato do metodo de "
                    "linearizacao, nao do dado -- use este ajuste (NLS, direto sobre Qe vs Ce) "
                    "como referencia mais robusta.")
        else:
            note = "Dados monotonicos, mas ajuste (linear ou NLS) ainda assim modesto (R2<0.7)."

        params = {"qmax": qmax_nls, "kl": kl_nls, "r2": r2_nls, "n_points": len(ce),
                   "method": "scipy.optimize.curve_fit, Qe=Qmax*KL*Ce/(1+KL*Ce)"}
        out.append(dict(
            treatment_id=treatment_id, temperature_C=temp_c, model="Langmuir (NLS)",
            qmax=qmax_nls, kl=kl_nls, rl=None, r2=r2_nls, params_json=json.dumps(params),
            notes=note, sheet_source="Isotermas VAGEM (recalculado a partir dos pontos brutos)",
        ))
        notes_by_key[(treatment_id, temp_c)] = note
    return out, notes_by_key


def extract_isotherm_palma(df, treatments):
    """
    Aba 'Parametros PALMA': tabela limpa com 6 modelos de isoterma (Langmuir,
    Freundlich, Temkin, Durbinin-Kaganer-Radushkevich, Harkin-Jura, Halsey), cada um
    com 3 tratamentos (Base/Acido/In natura) x 3 temperaturas (15/30/45 C).
    Layout por bloco de tratamento: coluna do rotulo do parametro + 3 colunas de
    valor (+1=15C, +2=30C, +3=45C). Colunas-base do rotulo: Base=1, Acido=6,
    In natura=11 (linhas conferidas manualmente contra os dumps da planilha).
    So o modelo Langmuir usa as colunas dedicadas qmax/kl aqui; os demais modelos
    tem parametros proprios sem coluna dedicada no schema e vao inteiros em
    params_json (nomes de variavel identicos aos da planilha).
    """
    treat_label_col = {"Base": 1, "Ácido": 6, "In natura": 11}
    temp_col_offset = {15.0: 1, 30.0: 2, 45.0: 3}
    models = [
        ("Langmuir", {"qmax": 5, "kl": 6}, 8),
        ("Freundlich", {"kf": 12, "one_over_nf": 13, "nf": 14}, 15),
        ("Temkin", {"AT": 19, "B": 20, "bT": 21}, 22),
        ("DKR", {"Qd": 26, "ADKR": 27, "E": 28}, 29),
        ("Harkin-Jura", {"one_over_AHJ": 33, "AHJ": 34, "BHJ": 35}, 36),
        ("Halsey", {"one_over_nH": 40, "nH": 41, "KH": 42}, 43),
    ]
    rows = []
    for model_name, param_rows, r2_row in models:
        for sheet_treat, label_col in treat_label_col.items():
            treat_name = "Acido" if sheet_treat == "Ácido" else sheet_treat
            for temp_c, off in temp_col_offset.items():
                col = label_col + off
                params = {pname: _num(df.iat[prow, col]) for pname, prow in param_rows.items()}
                r2 = _num(df.iat[r2_row, col])
                params["r2"] = r2
                if all(v is None for v in params.values()):
                    continue
                rows.append(dict(
                    treatment_id=treatments[("Palma", treat_name)], temperature_C=temp_c, model=model_name,
                    qmax=params.get("qmax"), kl=params.get("kl"), rl=None, r2=r2,
                    params_json=json.dumps(params), notes=None, sheet_source="Parâmetros PALMA",
                ))
    return rows


def build():
    global NEG_CE_TRUNCATED
    xlsx_path = fetch_xlsx(force=True)
    xls = pd.ExcelFile(xlsx_path)

    # ---- blindagem: falha alto e cedo se a planilha do Drive mudou de layout ----
    validate.run_pre_parse_checks(xls)

    # ---- constantes de calibracao / C0 por aba (ver docstring sobre o C0 legado) ----
    C0_PH_SHARED = 234.657926          # pH Vagem (Q5) e pH PALMA (P4) -- mesmo lote de estoque
    C0_MASSA_VAGEM_CORRETO = 223.334923   # Massa Vagem, celula propria Q6
    C0_MASSA_PALMA = 208.734207           # Massa PALMA, celula propria Q5
    C0_TEMPO_VAGEM = 201.582837            # Tempo (Vagem) e Tempo (Vagem 2), calibracao topo da aba
    C0_TEMPO_PALMA = 201.582837            # Tempo PALMA, U3 -- mesmo valor de calibracao

    blocks = []  # cada item: (material, treatment, experiment_type, sheet, rows_kwargs)

    # =========================================================== pH ===========================================================
    df = xls.parse("pH Vagem", header=None)
    blocks.append(("Vagem", "Acido", "pH", "pH Vagem", dict(
        rows=extract_rows(df, range(8, 13), ph_col=2, mass_cols=(3, 4), ce_col=11,
                           c0=C0_PH_SHARED, fixed=dict(tempo_min=1440, temperatura_C=25),
                           obs="tempo_min e temperatura_C assumidos (equilibrio 24h, temp ambiente)"))))
    blocks.append(("Vagem", "Base", "pH", "pH Vagem", dict(
        rows=extract_rows(df, range(20, 25), ph_col=2, mass_cols=(3, 4), ce_col=11,
                           c0=C0_PH_SHARED, fixed=dict(tempo_min=1440, temperatura_C=25),
                           obs="tempo_min e temperatura_C assumidos (equilibrio 24h, temp ambiente)"))))

    df = xls.parse("pH PALMA", header=None)
    blocks.append(("Palma", "Acido", "pH", "pH PALMA", dict(
        rows=extract_rows(df, range(7, 12), ph_col=0, mass_cols=(2, 3), ce_col=10,
                           c0=C0_PH_SHARED, fixed=dict(tempo_min=1440, temperatura_C=25),
                           obs="tempo_min e temperatura_C assumidos (equilibrio 24h, temp ambiente)"))))
    blocks.append(("Palma", "Base", "pH", "pH PALMA", dict(
        rows=extract_rows(df, range(34, 39), ph_col=0, mass_cols=(2, 3), ce_col=10,
                           c0=C0_PH_SHARED, fixed=dict(tempo_min=1440, temperatura_C=25),
                           obs="tempo_min e temperatura_C assumidos (equilibrio 24h, temp ambiente)"))))

    # ========================================================= Massa ==========================================================
    df = xls.parse("Massa Vagem", header=None)
    blocks.append(("Vagem", "Acido", "massa", "Massa Vagem", dict(
        rows=extract_rows(df, range(12, 16), ph_col=None, mass_cols=(3, 4), ce_col=11,
                           c0=C0_MASSA_VAGEM_CORRETO, fixed=dict(ph=10, tempo_min=1440, temperatura_C=25),
                           obs="C0 correto desta aba (Q6); pH=10 fixo"))))
    blocks.append(("Vagem", "Base", "massa", "Massa Vagem", dict(
        rows=extract_rows(df, range(24, 28), ph_col=None, mass_cols=(3, 4), ce_col=11,
                           c0=C0_MASSA_VAGEM_CORRETO, fixed=dict(ph=6, tempo_min=1440, temperatura_C=25),
                           obs="C0 correto desta aba (Q6); pH=6 fixo"))))

    df = xls.parse("Massa PALMA", header=None)
    blocks.append(("Palma", "Acido", "massa", "Massa PALMA", dict(
        rows=extract_rows(df, range(12, 19), ph_col=None, mass_cols=(3, 4), ce_col=11,
                           c0=C0_MASSA_PALMA, fixed=dict(ph=10, tempo_min=1440, temperatura_C=25),
                           obs="pH=10 fixo (anotado na aba)"))))
    blocks.append(("Palma", "Base", "massa", "Massa PALMA", dict(
        rows=extract_rows(df, range(27, 34), ph_col=None, mass_cols=(3, 4), ce_col=11,
                           c0=C0_MASSA_PALMA, fixed=dict(ph=6, tempo_min=1440, temperatura_C=25),
                           obs="pH=6 fixo (anotado na aba)"))))

    # ========================================================= Tempo ==========================================================
    df = xls.parse("Tempo (Vagem)", header=None)
    blocks.append(("Vagem", "Acido", "tempo", "Tempo (Vagem)", dict(
        rows=extract_rows(df, range(32, 46), ph_col=None, mass_cols=(3, 4), ce_col=11, tempo_col=2,
                           c0=C0_TEMPO_VAGEM, fixed=dict(ph=6, temperatura_C=25),
                           obs="pH=6 fixo (anotado na aba); temperatura_C assumida (ambiente)"))))
    blocks.append(("Vagem", "Base", "tempo", "Tempo (Vagem)", dict(
        rows=extract_rows(df, range(10, 24), ph_col=None, mass_cols=(3, 4), ce_col=11, tempo_col=2,
                           c0=C0_TEMPO_VAGEM, fixed=dict(ph=10, temperatura_C=25),
                           obs="pH=10 fixo (anotado na aba); temperatura_C assumida (ambiente)"))))

    df = xls.parse("Tempo (Vagem 2)", header=None)
    blocks.append(("Vagem", "Acido", "tempo", "Tempo (Vagem 2)", dict(
        rows=extract_rows(df, range(33, 47), ph_col=None, mass_cols=(2, 3), ce_col=10, tempo_col=1,
                           c0=C0_TEMPO_VAGEM, fixed=dict(ph=10, temperatura_C=25),
                           obs="pH=10 fixo (anotado na aba); temperatura_C assumida (ambiente)"))))
    blocks.append(("Vagem", "Base", "tempo", "Tempo (Vagem 2)", dict(
        rows=extract_rows(df, range(11, 25), ph_col=None, mass_cols=(2, 3), ce_col=10, tempo_col=1,
                           c0=C0_TEMPO_VAGEM, fixed=dict(ph=6, temperatura_C=25),
                           obs="pH=6 fixo (anotado na aba); temperatura_C assumida (ambiente)"))))

    df = xls.parse("Tempo PALMA", header=None)
    blocks.append(("Palma", "Acido", "tempo", "Tempo PALMA", dict(
        rows=extract_rows(df, range(7, 21), ph_col=None, mass_cols=(2, 3), ce_col=10, tempo_col=1,
                           c0=C0_TEMPO_PALMA, fixed=dict(ph=10, temperatura_C=25),
                           obs="pH=10 fixo (anotado na aba); temperatura_C assumida (ambiente)"))))
    blocks.append(("Palma", "Base", "tempo", "Tempo PALMA", dict(
        rows=extract_rows(df, range(29, 43), ph_col=None, mass_cols=(2, 3), ce_col=10, tempo_col=1,
                           c0=C0_TEMPO_PALMA, fixed=dict(ph=6, temperatura_C=25),
                           obs="pH=6 fixo (anotado na aba); temperatura_C assumida (ambiente)"))))

    # ====================================================== Conc. + Temp =======================================================
    df = xls.parse("Conc. e temp VAGEM", header=None)
    ct_vagem_acido = [(range(23, 27), 15), (range(51, 55), 30), (range(78, 82), 45)]
    ct_vagem_base = [(range(14, 18), 15), (range(43, 47), 30), (range(70, 74), 45)]
    for rows, temp_c in ct_vagem_acido:
        blocks.append(("Vagem", "Acido", "conc_temp", "Conc. e temp VAGEM", dict(
            rows=extract_rows(df, rows, ph_col=None, mass_cols=(2, 3, 4), ce_col=14, conc_col=1,
                               c0=None, fixed=dict(ph=6, tempo_min=1440, temperatura_C=temp_c),
                               obs="pH=6 IMPUTADO (isoterma sem controle de pH registrado); tempo_min assumido (equilibrio 24h); C0=concentracao nominal do ponto"))))
    for rows, temp_c in ct_vagem_base:
        blocks.append(("Vagem", "Base", "conc_temp", "Conc. e temp VAGEM", dict(
            rows=extract_rows(df, rows, ph_col=None, mass_cols=(2, 3, 4), ce_col=14, conc_col=1,
                               c0=None, fixed=dict(ph=6, tempo_min=1440, temperatura_C=temp_c),
                               obs="pH=6 IMPUTADO (isoterma sem controle de pH registrado); tempo_min assumido (equilibrio 24h); C0=concentracao nominal do ponto"))))

    df = xls.parse("Conc. e temp PALMA", header=None)
    ct_palma_acido = [(range(24, 28), 15), (range(60, 64), 30), (range(88, 92), 45)]
    ct_palma_base = [(range(14, 18), 15), (range(52, 56), 30), (range(80, 84), 45)]
    for rows, temp_c in ct_palma_acido:
        blocks.append(("Palma", "Acido", "conc_temp", "Conc. e temp PALMA", dict(
            rows=extract_rows(df, rows, ph_col=None, mass_cols=(2, 3), ce_col=10, conc_col=1,
                               c0=None, fixed=dict(ph=6, tempo_min=1440, temperatura_C=temp_c),
                               obs="pH=6 IMPUTADO (isoterma sem controle de pH registrado); tempo_min assumido (equilibrio 24h); C0=concentracao nominal do ponto"))))
    for rows, temp_c in ct_palma_base:
        blocks.append(("Palma", "Base", "conc_temp", "Conc. e temp PALMA", dict(
            rows=extract_rows(df, rows, ph_col=None, mass_cols=(2, 3), ce_col=10, conc_col=1,
                               c0=None, fixed=dict(ph=6, tempo_min=1440, temperatura_C=temp_c),
                               obs="pH=6 IMPUTADO (isoterma sem controle de pH registrado); tempo_min assumido (equilibrio 24h); C0=concentracao nominal do ponto"))))

    # ------------------------------------------------------------------------------------------------------------------------
    # Escreve num arquivo TEMPORARIO -- o banco de producao (DB_PATH) so e substituido
    # depois que validate.run_post_build_checks() passar (ver fim da funcao). Assim,
    # se a planilha do Drive mudou de layout e os dados saem errados, o dashboard/ML
    # continuam servindo o ultimo banco bom conhecido em vez de dados corrompidos.
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(TMP_DB_PATH):
        os.remove(TMP_DB_PATH)
    conn = sqlite3.connect(TMP_DB_PATH)
    conn.executescript(SCHEMA)

    materials = {}
    for name in ("Palma", "Vagem"):
        cur = conn.execute("INSERT INTO materials(name) VALUES (?)", (name,))
        materials[name] = cur.lastrowid

    treatments = {}
    for mat_name, mat_id in materials.items():
        # "In natura" (fibra sem tratamento quimico) so tem dados nas abas de isoterma
        # (Isotermas VAGEM / Parametros PALMA) -- nao tem experiment_runs OFAT proprios
        # no recorte atual (pH/Massa/Tempo/Conc+Temp so cobrem Acido/Base).
        for tname in ("Acido", "Base", "In natura"):
            cur = conn.execute("INSERT INTO treatments(material_id, name) VALUES (?,?)", (mat_id, tname))
            treatments[(mat_name, tname)] = cur.lastrowid

    conn.executemany(
        "INSERT INTO calibration_curves(material_id, sheet_source, slope, intercept, equation_text, notes) VALUES (?,?,?,?,?,?)",
        [
            (materials["Vagem"], "Curva Vagem", 0.1678, 0.0244, "Y=0,1678x+0,0244",
             "Curva principal (pH/Massa/Tempo), compartilhada com Palma"),
            (materials["Vagem"], "Conc. e temp VAGEM", 0.0772, 0.0243, "Y=0,0772x+0,0243",
             "Curva especifica do experimento de concentracao/temperatura (faixa mais diluida)"),
            (materials["Palma"], "Curva de Calibração", 0.1678, 0.0244, "Y=0,1678x+0,0244",
             "Curva principal, identica a usada em Vagem (mesmo lote de reagente)"),
        ],
    )

    n_runs_total = 0
    counts = {}
    for mat_name, treat_name, exp_type, sheet, kw in blocks:
        rows = kw["rows"]
        treatment_id = treatments[(mat_name, treat_name)]
        cur = conn.execute(
            "INSERT INTO experiments(treatment_id, experiment_type, sheet_source, volume_L, notes) VALUES (?,?,?,?,?)",
            (treatment_id, exp_type, sheet, V_L, None),
        )
        exp_id = cur.lastrowid
        for row in rows:
            conn.execute(
                """INSERT INTO experiment_runs
                   (experiment_id, ph, massa_g, tempo_min, conc_inicial_mgL, temperatura_C,
                    ce_mgL, qe_mgg, remocao_pct, is_assumed_tempo, is_assumed_temp, is_assumed_ph, obs)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (exp_id, row["ph"], row["massa_g"], row["tempo_min"], row["conc_inicial_mgL"],
                 row["temperatura_C"], row["ce_mgL"], row["qe_mgg"], row["remocao_pct"],
                 0, 0, 0, row["obs"]),
            )
            n_runs_total += 1
        key = (mat_name, treat_name)
        counts[key] = counts.get(key, 0) + len(rows)

    # ---- isotherm_params (Langmuir p/ Vagem via 'Isotermas VAGEM'; 6 modelos p/ Palma via 'Parâmetros PALMA') ----
    df_iso_vagem = xls.parse("Isotermas VAGEM", header=None)
    iso_rows = []
    iso_rows += extract_isotherm_langmuir_vagem(df_iso_vagem, treatments)
    iso_rows += extract_isotherm_palma(xls.parse("Parâmetros PALMA", header=None), treatments)

    # Pontos brutos (Ci/Ce/Qe) de Vagem, para permitir reanalise/re-ajuste (ver isotherm_raw_points)
    raw_points = extract_isotherm_vagem_raw_points(df_iso_vagem, treatments)

    # Reajuste por regressao nao-linear direta (mais robusto que a linearizacao Ce/Qe vs Ce da
    # planilha) + notas de qualidade, anexadas tambem ao model='Langmuir' original do mesmo bloco.
    nls_rows, notes_by_key = refit_langmuir_nls(raw_points)
    for row in iso_rows:
        if row["model"] == "Langmuir":
            key = (row["treatment_id"], row["temperature_C"])
            row["notes"] = notes_by_key.get(key)
    iso_rows += nls_rows

    for row in iso_rows:
        conn.execute(
            """INSERT INTO isotherm_params
               (treatment_id, temperature_C, model, qmax, kl, rl, r2, params_json, notes, sheet_source)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (row["treatment_id"], row["temperature_C"], row["model"], row["qmax"], row["kl"],
             row["rl"], row["r2"], row["params_json"], row["notes"], row["sheet_source"]),
        )

    for row in raw_points:
        conn.execute(
            """INSERT INTO isotherm_raw_points
               (treatment_id, temperature_C, ci_mgL, ce_mgL, qe_mgg, sheet_source)
               VALUES (?,?,?,?,?,?)""",
            (row["treatment_id"], row["temperature_C"], row["ci_mgL"], row["ce_mgL"], row["qe_mgg"],
             row["sheet_source"]),
        )

    conn.commit()

    # ---- blindagem: valida o banco recem-construido ANTES de promove-lo a producao ----
    try:
        validate.run_post_build_checks(conn, counts)
    except validate.ValidationError:
        conn.close()
        if os.path.exists(TMP_DB_PATH):
            os.remove(TMP_DB_PATH)
        print("VALIDACAO FALHOU -- banco de producao NAO foi alterado (ultimo banco bom conhecido preservado).")
        raise
    conn.close()

    os.replace(TMP_DB_PATH, DB_PATH)  # atomico (POSIX): so troca o banco de producao se chegou ate aqui

    print(f"Concentracoes de equilibrio negativas truncadas para 0: {NEG_CE_TRUNCATED}")
    print(f"Total de experiment_runs inseridos: {n_runs_total}")
    for k, v in sorted(counts.items()):
        print(f"  {k[0]:6s} / {k[1]:6s}: {v} linhas")
    print(f"Total de isotherm_params inseridos: {len(iso_rows)} (incl. {len(nls_rows)} Langmuir (NLS))")
    print(f"Total de isotherm_raw_points inseridos: {len(raw_points)}")
    print("Validacao: OK (sheets, celulas-ancora, contagens e faixas plausiveis conferidas)")
    print(f"Banco salvo em: {DB_PATH}")


if __name__ == "__main__":
    try:
        build()
    except validate.ValidationError as e:
        print("\n" + "=" * 70)
        print("ETL ABORTADO -- validacao de blindagem falhou:")
        print(str(e))
        print("=" * 70)
        sys.exit(1)
