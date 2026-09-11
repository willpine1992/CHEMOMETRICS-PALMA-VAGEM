"""
Blindagem do ETL: confere que a planilha baixada do Drive ainda tem a estrutura que
o parser (build_database.py) espera -- nomes de aba, celulas-ancora usadas para
posicionar os indices fixos de linha/coluna, contagens de linha por bloco -- e que
os dados extraidos caem em faixas fisicamente plausiveis.

Filosofia: o parser usa indices numericos fixos (ex. "pH Vagem, linhas 8-12, coluna
2") descobertos manualmente inspecionando a planilha. Se alguem reordenar linhas,
renomear uma aba, ou inserir uma coluna no Drive, o parser pode continuar rodando
SEM lancar nenhum erro Python -- so vai ler os numeros errados. Isso e o pior tipo
de falha (silenciosa). As checagens aqui existem para transformar esse tipo de
problema em uma falha BARULHENTA (ValidationError), antes que dados errados
substituam o banco de producao -- ver como build() usa isso (escreve num arquivo
temporario e so promove pra DB_PATH se validate.run_all() passar).

Se voce mudar a planilha de proposito (nova linha de dado, aba renomeada etc.) e as
contagens abaixo pararem de bater, atualize as constantes EXPECTED_* aqui -- isso e
esperado e faz parte de manter o ETL em dia com a fonte.
"""


class ValidationError(Exception):
    pass


EXPECTED_SHEETS = [
    "Curva de Calibração", "pH PALMA", "Massa PALMA", "Tempo PALMA", "Conc. e temp PALMA",
    "Isotermas PALMA", "Ce x Qe PALMA", "Parâmetros PALMA", "pH Vagem", "Massa Vagem",
    "Tempo (Vagem)", "Tempo (Vagem 2)", "tempo ph 6 e 10", "Curva Vagem",
    "Conc. e temp VAGEM", "Isotermas VAGEM", "Ce x Qe VAGEM", "Parâmetros VAGEM",
]

# Celulas cujo TEXTO ancora os indices fixos de linha/coluna do parser. Se o texto
# mudou, os offsets hardcoded deixaram de ser confiaveis mesmo sem erro Python.
HEADER_ANCHORS = [
    ("pH Vagem", 6, 1, "Valores de pH"),
    ("pH Vagem", 7, 2, "Exp."),
    ("Massa Vagem", 10, 1, "Valores das massas"),
    ("Tempo (Vagem)", 8, 2, "Tempo (min)"),
    ("Isotermas VAGEM", 0, 1, "Qmax"),
    ("Isotermas VAGEM", 2, 1, "Langmuir"),
    ("Isotermas VAGEM", 3, 1, "Base"),
    ("Isotermas VAGEM", 3, 5, "Ácido"),
    ("Isotermas VAGEM", 3, 9, "In natura"),
    ("Isotermas VAGEM", 4, 0, "Ci"),
    ("Parâmetros PALMA", 3, 1, "Langmuir"),
    ("Parâmetros PALMA", 4, 1, "Parâmetros"),
    ("Parâmetros PALMA", 38, 1, "Halsey"),
]

# "Golden counts" confirmados manualmente na ultima extracao validada -- qualquer
# desvio pede investigacao (linha nova/removida na planilha, ou parser desalinhado).
EXPECTED_RUN_COUNTS = {
    ("Palma", "Acido"): 38, ("Palma", "Base"): 38,
    ("Vagem", "Acido"): 49, ("Vagem", "Base"): 49,
}
EXPECTED_MATERIALS = 2
EXPECTED_TREATMENTS = 6
EXPECTED_CALIBRATION_CURVES = 3
EXPECTED_ISOTHERM_PARAMS_TOTAL = 72
EXPECTED_RAW_POINTS_TOTAL = 36

# Faixas fisicamente plausiveis para as colunas de experiment_runs -- nao sao limites
# "corretos" cientificamente, so um cinto de seguranca contra erro grosseiro de
# parsing (ex.: ler a coluna errada e pegar um numero de 3 digitos onde deveria ser pH).
PLAUSIBLE_RANGES = {
    "ph": (0, 14),
    "massa_g": (0.0001, 5),
    "tempo_min": (0, 4000),
    "conc_inicial_mgL": (0, 2000),
    "temperatura_C": (0, 60),
    "qe_mgg": (-5, 1000),
    "remocao_pct": (-60, 150),
}


def check_sheets_present(xls):
    missing = [s for s in EXPECTED_SHEETS if s not in xls.sheet_names]
    if missing:
        raise ValidationError(
            f"Abas esperadas nao encontradas na planilha baixada do Drive: {missing}. "
            "A planilha pode ter sido renomeada/reestruturada -- abortando antes de "
            "escrever no banco de producao."
        )


def check_header_anchors(xls):
    failures = []
    for sheet, row, col, expected in HEADER_ANCHORS:
        df = xls.parse(sheet, header=None)
        try:
            actual = str(df.iat[row, col])
        except IndexError:
            actual = "<fora dos limites da planilha>"
        if expected.lower() not in actual.lower():
            failures.append(f"  aba='{sheet}' celula=({row},{col}) esperado~'{expected}' encontrado='{actual}'")
    if failures:
        raise ValidationError(
            "Celulas-ancora nao batem com o esperado -- o layout da planilha no Drive "
            "provavelmente mudou e os indices fixos do parser ficaram desalinhados:\n"
            + "\n".join(failures)
        )


def check_experiment_run_counts(counts):
    problems = []
    for key, expected in EXPECTED_RUN_COUNTS.items():
        got = counts.get(key, 0)
        if got != expected:
            problems.append(f"  {key[0]}/{key[1]}: esperado {expected} linhas, obtido {got}")
    if problems:
        raise ValidationError(
            "Contagem de experiment_runs diferente do esperado (linha adicionada/removida "
            "na planilha, ou parser desalinhado):\n" + "\n".join(problems) +
            "\nSe a mudanca foi intencional, atualize EXPECTED_RUN_COUNTS em etl/validate.py."
        )


def check_totals(conn):
    def count(table):
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    problems = []
    checks = [
        ("materials", EXPECTED_MATERIALS),
        ("treatments", EXPECTED_TREATMENTS),
        ("calibration_curves", EXPECTED_CALIBRATION_CURVES),
        ("isotherm_params", EXPECTED_ISOTHERM_PARAMS_TOTAL),
        ("isotherm_raw_points", EXPECTED_RAW_POINTS_TOTAL),
    ]
    for table, expected in checks:
        got = count(table)
        if got != expected:
            problems.append(f"  {table}: esperado {expected}, obtido {got}")
    if problems:
        raise ValidationError("Contagens totais do banco fora do esperado:\n" + "\n".join(problems))


def check_plausible_ranges(conn):
    problems = []
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        n = conn.execute(
            f"SELECT COUNT(*) FROM experiment_runs WHERE {col} IS NOT NULL AND ({col} < ? OR {col} > ?)",
            (lo, hi),
        ).fetchone()[0]
        if n:
            problems.append(f"  {n} linha(s) de experiment_runs com {col} fora da faixa plausivel [{lo},{hi}]")

    n_r2 = conn.execute("SELECT COUNT(*) FROM isotherm_params WHERE r2 IS NOT NULL AND r2 > 1.0001").fetchone()[0]
    if n_r2:
        problems.append(f"  {n_r2} linha(s) de isotherm_params com R² > 1 (matematicamente impossivel)")

    n_neg_qmax = conn.execute("SELECT COUNT(*) FROM isotherm_params WHERE qmax IS NOT NULL AND qmax <= 0").fetchone()[0]
    if n_neg_qmax:
        problems.append(f"  {n_neg_qmax} linha(s) de isotherm_params com Qmax <= 0 (nao-fisico)")

    if problems:
        raise ValidationError(
            "Valores fora de faixa plausivel no banco recem-construido:\n" + "\n".join(problems)
        )


def run_pre_parse_checks(xls):
    """Roda ANTES de qualquer parsing/insercao -- falha rapido, sem gastar tempo."""
    check_sheets_present(xls)
    check_header_anchors(xls)


def run_post_build_checks(conn, counts):
    """Roda no banco recem-construido (ainda no arquivo temporario), antes de
    promove-lo a DB_PATH."""
    check_experiment_run_counts(counts)
    check_totals(conn)
    check_plausible_ranges(conn)
