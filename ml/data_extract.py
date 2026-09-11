"""
Extracao do dataset de ML a partir de "ARTIGO VAGEN/Isotermas Vagem.xlsx".

Decisao de escopo (documentada no relatorio): o workbook tem 3 tratamentos de fibra
(Acido, Base, In Natura) testados em experimentos OFAT (um fator por vez) separados
para pH, massa, tempo e concentracao/temperatura. Combinar os 3 tratamentos num unico
modelo confundiria o efeito do tratamento com os efeitos experimentais pedidos pelo
usuario (pH, massa, tempo, concentracao, temperatura - apenas 5 features, sem
"tratamento"). Como o tratamento "Acido" aparece de forma completa e consistente em
TODAS as 5 fontes de dados (pH, massa, tempo x2, conc/temp) e sozinho ja fornece
dataset >= 25-30 linhas exigido, filtramos para "Fibra Vagem/Vagem tratada com Acido".

Qe (mg/g) e %Remocao sao SEMPRE recalculados aqui pela formula padrao
Qe = (C0 - Ce) * V / m , %R = (C0-Ce)/C0*100
em vez de usar as colunas "QE (mg/g)" ja existentes no Excel, porque essas colunas
mostraram inconsistencia interna (ex.: aba "Tempo (Vagem)" tem QE negativo/fora de
ordem de grandeza quando comparado ao recalculo manual - provavel erro de formula
na planilha original). Isso garante metodologia uniforme em todas as linhas do
dataset final. C0 (concentracao inicial real) e Ce sao tomados a partir das colunas
de concentracao ja convertidas via curva de calibracao (Absorbancia -> ppm) that
already exist in each sheet; C0 é a concentraçao real do lote de estoque usado em
cada bloco de experimentos (validado batendo com a coluna "% Remocao"/"Média %" do
proprio Excel - ver relatorio).
"""
import numpy as np
import pandas as pd

XLSX = "/Users/macbookpro/Documents/POSDOC - MAC/QUIMIOINFORMÁTICA/ARTIGO VAGEN/Isotermas Vagem.xlsx"
OUT_CSV = "/Users/macbookpro/Documents/POSDOC - MAC/QUIMIOINFORMÁTICA/DASHBOARD-VAGEM/ml/data/vagem_ml_dataset.csv"

V_L = 0.02  # volume da solucao (L), constante em todos os experimentos Vagem (nota "V=0.02" nas planilhas)
NEG_CE_TRUNCATED = 0  # contador global


def qe_and_remocao(c0, ce, m_g):
    global NEG_CE_TRUNCATED
    ce_clipped = ce.copy() if hasattr(ce, "copy") else ce
    if np.isscalar(ce):
        if ce < 0:
            NEG_CE_TRUNCATED += 1
            ce_clipped = 0.0
    qe = (c0 - ce_clipped) * V_L / m_g
    remocao = (c0 - ce_clipped) / c0 * 100.0
    return qe, remocao


rows = []
xls = pd.ExcelFile(XLSX)

# ---------------------------------------------------------------- pH Vagem (Acido)
df = xls.parse("pH Vagem", header=None)
C0_PH_MASSA = 234.657926  # calibracao do lote de estoque usado nas abas pH Vagem / Massa Vagem
for r in range(8, 13):
    ph = df.iat[r, 2]
    m1, m2 = df.iat[r, 3], df.iat[r, 4]
    ce = df.iat[r, 11]
    m_g = np.mean([m1, m2])
    qe, remocao = qe_and_remocao(C0_PH_MASSA, ce, m_g)
    rows.append(dict(
        ph=ph, massa_g=m_g, tempo_min=1440, conc_inicial_mgL=C0_PH_MASSA,
        temperatura_C=25, ce_mgL=max(ce, 0), qe_mgg=qe, remocao_pct=remocao,
        origem_aba="pH Vagem", tratamento="Acido",
        obs="tempo_min e temperatura_C assumidos (equilibrio 24h, temp ambiente) - nao informados na aba",
    ))

# ---------------------------------------------------------------- Massa Vagem (Acido, pH=10)
df = xls.parse("Massa Vagem", header=None)
for r in range(12, 16):
    massa_mg = df.iat[r, 1]
    m1, m2 = df.iat[r, 3], df.iat[r, 4]
    ce = df.iat[r, 11]
    m_g = np.mean([m1, m2])
    qe, remocao = qe_and_remocao(C0_PH_MASSA, ce, m_g)
    rows.append(dict(
        ph=10, massa_g=m_g, tempo_min=1440, conc_inicial_mgL=C0_PH_MASSA,
        temperatura_C=25, ce_mgL=max(ce, 0), qe_mgg=qe, remocao_pct=remocao,
        origem_aba="Massa Vagem", tratamento="Acido",
        obs="pH=10 fixo (anotado na aba); tempo_min e temperatura_C assumidos",
    ))

# ---------------------------------------------------------------- Tempo (Vagem) (Acido, pH=6)
C0_TEMPO = 201.582837
df = xls.parse("Tempo (Vagem)", header=None)
for r in range(32, 46):
    tempo = df.iat[r, 2]
    m1, m2 = df.iat[r, 3], df.iat[r, 4]
    ce = df.iat[r, 11]
    m_g = np.mean([m1, m2])
    qe, remocao = qe_and_remocao(C0_TEMPO, ce, m_g)
    rows.append(dict(
        ph=6, massa_g=m_g, tempo_min=tempo, conc_inicial_mgL=C0_TEMPO,
        temperatura_C=25, ce_mgL=max(ce, 0), qe_mgg=qe, remocao_pct=remocao,
        origem_aba="Tempo (Vagem)", tratamento="Acido",
        obs="pH=6 fixo (anotado na aba); temperatura_C assumida (ambiente)",
    ))

# ---------------------------------------------------------------- Tempo (Vagem 2) (Acido, pH=10)
df = xls.parse("Tempo (Vagem 2)", header=None)
for r in range(33, 47):
    tempo = df.iat[r, 1]
    m1, m2 = df.iat[r, 2], df.iat[r, 3]
    ce = df.iat[r, 10]
    m_g = np.mean([m1, m2])
    qe, remocao = qe_and_remocao(C0_TEMPO, ce, m_g)
    rows.append(dict(
        ph=10, massa_g=m_g, tempo_min=tempo, conc_inicial_mgL=C0_TEMPO,
        temperatura_C=25, ce_mgL=max(ce, 0), qe_mgg=qe, remocao_pct=remocao,
        origem_aba="Tempo (Vagem 2)", tratamento="Acido",
        obs="pH=10 fixo (anotado na aba); temperatura_C assumida (ambiente)",
    ))

# ---------------------------------------------------------------- Conc. e temp VAGEM (Acido)
df = xls.parse("Conc. e temp VAGEM", header=None)
# blocos Acido: (linha header, [linhas de dados], temperatura)
acido_blocks = [(22, range(23, 27), 15), (50, range(51, 55), 30), (77, range(78, 82), 45)]
for header_row, data_rows, temp_c in acido_blocks:
    for r in data_rows:
        conc_nominal = df.iat[r, 1]
        masses = [df.iat[r, c] for c in (2, 3, 4) if pd.notna(df.iat[r, c])]
        m_g = float(np.mean(masses))
        ce = df.iat[r, 14]
        qe, remocao = qe_and_remocao(conc_nominal, ce, m_g)
        rows.append(dict(
            ph=6, massa_g=m_g, tempo_min=1440, conc_inicial_mgL=conc_nominal,
            temperatura_C=temp_c, ce_mgL=max(ce, 0), qe_mgg=qe, remocao_pct=remocao,
            origem_aba="Conc. e temp VAGEM", tratamento="Acido",
            obs="pH=6 IMPUTADO (isoterma sem controle de pH registrado); tempo_min assumido (equilibrio 24h)",
        ))

data = pd.DataFrame(rows)
print(f"Linhas extraidas (antes de limpeza): {len(data)}")
print(f"Concentracoes de equilibrio negativas truncadas para 0: {NEG_CE_TRUNCATED}")

data.to_csv(OUT_CSV, index=False)
print(f"Salvo em: {OUT_CSV}")
print(data.describe(include="all"))
