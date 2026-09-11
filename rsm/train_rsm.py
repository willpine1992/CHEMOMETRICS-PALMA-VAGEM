"""
Otimizacao Multivariada de Processos (DoE & RSM) para adsorcao da fibra de Vagem
(tratamento Acido) -- mesma fonte SQL (db/biosorcao.db) e mesmo recorte de dados
(material='Vagem', tratamento='Acido') usados pelo pipeline de ML/XAI em ml/.

LEIA ANTES DE USAR OS RESULTADOS: os dados vem de experimentos "um fator por vez"
(OFAT: pH, massa, tempo, concentracao+temperatura variados separadamente), NAO de
um Central Composite Design (CCD) ou Box-Behnken Design (BBD) desenhado a priori.
RSM classico pressupoe um design ortogonal/rotacionavel; aplicado retrospectivamente
a dados OFAT, a superficie de resposta ainda descreve validamente a relacao entre
fatores e resposta NESTE conjunto de dados, mas sem as garantias de poder/eficiencia
de um design apropriado -- e varias interacoes sao literalmente inestimaveis por
colinearidade estrutural do OFAT (ex.: massa e tempo tem correlacao de -0.89 entre
si nos dados, porque cada bloco de experimento variava um enquanto fixava o outro).

Um modelo quadratico completo (5 lineares + 5 quadraticos + 10 interacoes = 21
termos) e LITERALMENTE SINGULAR neste dataset (matriz rank-deficiente, numero de
condicao ~1e17, "SingularMatrixWarning"). Selecao de termos foi feita via VIF
(Variance Inflation Factor): das 10 interacoes possiveis, 7 tem VIF entre ~1500 e
>1e15 quando adicionadas ao modelo principal+quadratico (sem suporte de dados) e
foram descartadas; sobraram 3 com VIF < 2 (bem-condicionadas):

    Y ~ pH + massa + tempo + conc + temp                     (5 termos lineares)
      + pH^2 + massa^2 + tempo^2 + conc^2 + temp^2            (5 termos quadraticos)
      + pH:massa + pH:tempo + conc:temp                       (3 interacoes estimaveis)

Modelo final: numero de condicao ~29, todos os VIFs < 12, R²=0.96 (Qe) / 0.90 (%R).
Ajustado para os dois alvos (Qe e %Remocao), como no pipeline de ML.
"""
import json
import os
import sqlite3

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.optimize import differential_evolution

DB_PATH = "/Users/macbookpro/Documents/POSDOC - MAC/QUIMIOINFORMÁTICA/DASHBOARD-VAGEM/db/biosorcao.db"
ARTIFACTS_DIR = "/Users/macbookpro/Documents/POSDOC - MAC/QUIMIOINFORMÁTICA/DASHBOARD-VAGEM/rsm/artifacts"

FEATURES = ["ph", "massa_g", "tempo_min", "conc_inicial_mgL", "temperatura_C"]
FEATURE_LABELS = {
    "ph": "pH", "massa_g": "Massa (g)", "tempo_min": "Tempo (min)",
    "conc_inicial_mgL": "Concentração inicial (mg/L)", "temperatura_C": "Temperatura (°C)",
}
# Interacoes com suporte de dados suficiente (VIF < 2 quando adicionadas ao modelo
# principal+quadratico) -- as outras 7 pares foram descartadas por colinearidade
# estrutural do OFAT (VIF entre ~1500 e >1e15). Ver docstring.
INTERACTIONS = [("ph", "massa_g"), ("ph", "tempo_min"), ("conc_inicial_mgL", "temperatura_C")]
DROPPED_INTERACTIONS = [
    ("ph", "conc_inicial_mgL"), ("ph", "temperatura_C"), ("massa_g", "tempo_min"),
    ("massa_g", "conc_inicial_mgL"), ("massa_g", "temperatura_C"),
    ("tempo_min", "conc_inicial_mgL"), ("tempo_min", "temperatura_C"),
]

SQL_QUERY = """
SELECT er.ph, er.massa_g, er.tempo_min, er.conc_inicial_mgL, er.temperatura_C,
       er.qe_mgg, er.remocao_pct
FROM experiment_runs er
JOIN experiments e ON er.experiment_id = e.id
JOIN treatments t ON e.treatment_id = t.id
JOIN materials m ON t.material_id = m.id
WHERE m.name = 'Vagem' AND t.name = 'Acido'
"""


def load_dataset():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(SQL_QUERY, conn)
    finally:
        conn.close()
    return df


def pretty_term_name(raw):
    mapping = {f + "_z": FEATURE_LABELS[f] for f in FEATURES}
    if raw.startswith("I(") and raw.endswith(" ** 2)"):
        key = raw[2:-6]
        return mapping.get(key, key) + "²"
    if ":" in raw:
        a, b = raw.split(":")
        return mapping.get(a, a) + " × " + mapping.get(b, b)
    return mapping.get(raw, raw)


def build_formula(target):
    zf = [f + "_z" for f in FEATURES]
    linear = " + ".join(zf)
    quad = " + ".join([f"I({f}**2)" for f in zf])
    inter = " + ".join([f"{a}_z:{b}_z" for a, b in INTERACTIONS])
    return f"{target} ~ {linear} + {quad} + {inter}"


def fit_rsm(dfz, target):
    model = smf.ols(build_formula(target), data=dfz).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    return model, anova


def anova_to_json(anova, model):
    rows = []
    for term in anova.index:
        if term == "Residual":
            continue
        rows.append(dict(
            term=pretty_term_name(term), raw_term=term,
            sum_sq=float(anova.loc[term, "sum_sq"]), df=float(anova.loc[term, "df"]),
            F=float(anova.loc[term, "F"]), p=float(anova.loc[term, "PR(>F)"]),
            coef=float(model.params.get(term, np.nan)), se=float(model.bse.get(term, np.nan)),
            significant=bool(anova.loc[term, "PR(>F)"] < 0.05),
        ))
    rows.sort(key=lambda r: r["p"])
    resid = anova.loc["Residual"]
    rows.append(dict(term="Resíduo", raw_term="Residual", sum_sq=float(resid["sum_sq"]),
                      df=float(resid["df"]), F=None, p=None, coef=None, se=None, significant=None))
    model_f = float(model.fvalue)
    model_p = float(model.f_pvalue)
    return rows, model_f, model_p


def pareto_json(model, t_crit):
    points = []
    for term, t_val in model.tvalues.items():
        if term == "Intercept":
            continue
        points.append(dict(term=pretty_term_name(term), abs_t=float(abs(t_val)), t=float(t_val),
                            significant=bool(abs(t_val) > t_crit)))
    points.sort(key=lambda p: -p["abs_t"])
    return {"points": points, "t_critical": float(t_crit)}


def predict_row(model, means, stds, values):
    row = {f + "_z": (values[f] - means[f]) / stds[f] for f in FEATURES}
    return float(model.predict(pd.DataFrame([row]))[0])


def response_surface(model, means, stds, x_feat, y_feat, bounds, fixed, n=30):
    xs = np.linspace(bounds[x_feat][0], bounds[x_feat][1], n)
    ys = np.linspace(bounds[y_feat][0], bounds[y_feat][1], n)
    z = []
    for yv in ys:
        row_vals = []
        for xv in xs:
            values = dict(fixed)
            values[x_feat] = xv
            values[y_feat] = yv
            row_vals.append(predict_row(model, means, stds, values))
        z.append(row_vals)
    return {
        "x": xs.tolist(), "y": ys.tolist(), "z": z,
        "x_feature": x_feat, "y_feature": y_feat,
        "x_label": FEATURE_LABELS[x_feat], "y_label": FEATURE_LABELS[y_feat],
        "fixed": fixed,
    }


def optimize_desirability(model_qe, means, stds, bounds, qe_observed_range, qe_weight=1.0, massa_weight=1.0):
    qe_lo, qe_hi = qe_observed_range
    massa_lo, massa_hi = bounds["massa_g"]

    def neg_desirability(x):
        values = {FEATURES[i]: x[i] for i in range(len(FEATURES))}
        qe_pred = predict_row(model_qe, means, stds, values)
        d_qe = np.clip((qe_pred - qe_lo) / (qe_hi - qe_lo), 0.0, 1.0)
        d_massa = np.clip((massa_hi - values["massa_g"]) / (massa_hi - massa_lo), 0.0, 1.0)
        d = (d_qe ** qe_weight * d_massa ** massa_weight) ** (1.0 / (qe_weight + massa_weight))
        return -d

    bnds = [bounds[f] for f in FEATURES]
    result = differential_evolution(neg_desirability, bnds, seed=42, tol=1e-10, maxiter=400, polish=True)
    x_opt = result.x
    values = {FEATURES[i]: float(x_opt[i]) for i in range(len(FEATURES))}
    qe_opt = predict_row(model_qe, means, stds, values)
    return {
        "settings": values, "qe_pred": qe_opt, "desirability": float(-result.fun),
        "qe_weight": qe_weight, "massa_weight": massa_weight,
        "qe_observed_range": {"min": qe_lo, "max": qe_hi},
    }


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df = load_dataset()
    n_rows = len(df)

    means = {f: float(df[f].mean()) for f in FEATURES}
    stds = {f: float(df[f].std()) for f in FEATURES}
    bounds = {f: (float(df[f].min()), float(df[f].max())) for f in FEATURES}

    dfz = df.copy()
    for f in FEATURES:
        dfz[f + "_z"] = (df[f] - means[f]) / stds[f]

    results = {}
    for target in ["qe_mgg", "remocao_pct"]:
        model, anova = fit_rsm(dfz, target)
        t_crit = float(stats.t.ppf(0.975, model.df_resid))
        anova_rows, model_f, model_p = anova_to_json(anova, model)
        pareto = pareto_json(model, t_crit)

        results[target] = dict(
            model=model, anova_rows=anova_rows, pareto=pareto,
            r2=float(model.rsquared), r2_adj=float(model.rsquared_adj),
            condition_number=float(model.condition_number),
            model_f=model_f, model_p=model_p,
            n_params=int(len(model.params)), df_resid=float(model.df_resid),
        )

        with open(f"{ARTIFACTS_DIR}/anova_{target}.json", "w") as fh:
            json.dump({
                "rows": anova_rows, "r2": results[target]["r2"], "r2_adj": results[target]["r2_adj"],
                "condition_number": results[target]["condition_number"],
                "model_f": model_f, "model_p": model_p,
                "n_obs": n_rows, "n_params": results[target]["n_params"],
                "df_resid": results[target]["df_resid"],
            }, fh, indent=2)
        with open(f"{ARTIFACTS_DIR}/pareto_{target}.json", "w") as fh:
            json.dump(pareto, fh, indent=2)

    # Superficie de resposta: pH x Massa (unica interacao estatisticamente significativa
    # para Qe, p=0.012 -- e tambem o par sugerido no pedido original). Outros fatores
    # fixados na media amostral (centro do design).
    qe_model = results["qe_mgg"]["model"]
    fixed_center = {f: means[f] for f in FEATURES if f not in ("ph", "massa_g")}
    surface_qe = response_surface(qe_model, means, stds, "ph", "massa_g", bounds, fixed_center)
    with open(f"{ARTIFACTS_DIR}/surface_qe.json", "w") as fh:
        json.dump(surface_qe, fh)

    rem_model = results["remocao_pct"]["model"]
    surface_rem = response_surface(rem_model, means, stds, "ph", "massa_g", bounds, fixed_center)
    with open(f"{ARTIFACTS_DIR}/surface_remocao.json", "w") as fh:
        json.dump(surface_rem, fh)

    # Otimizacao por desirability: maximizar Qe previsto minimizando massa de adsorvente,
    # dentro dos limites observados (sem extrapolar para fora do espaco experimental).
    qe_range = (float(df["qe_mgg"].min()), float(df["qe_mgg"].max()))
    optimum = optimize_desirability(qe_model, means, stds, bounds, qe_range)
    with open(f"{ARTIFACTS_DIR}/optimum.json", "w") as fh:
        json.dump(optimum, fh, indent=2)

    summary = {
        "n_rows": n_rows,
        "features": FEATURES, "feature_labels": FEATURE_LABELS,
        "interactions_kept": [f"{FEATURE_LABELS[a]} × {FEATURE_LABELS[b]}" for a, b in INTERACTIONS],
        "interactions_dropped": [f"{FEATURE_LABELS[a]} × {FEATURE_LABELS[b]}" for a, b in DROPPED_INTERACTIONS],
        "bounds": bounds,
        "qe": {"r2": results["qe_mgg"]["r2"], "r2_adj": results["qe_mgg"]["r2_adj"],
               "condition_number": results["qe_mgg"]["condition_number"],
               "model_f": results["qe_mgg"]["model_f"], "model_p": results["qe_mgg"]["model_p"]},
        "remocao": {"r2": results["remocao_pct"]["r2"], "r2_adj": results["remocao_pct"]["r2_adj"],
                    "condition_number": results["remocao_pct"]["condition_number"],
                    "model_f": results["remocao_pct"]["model_f"], "model_p": results["remocao_pct"]["model_p"]},
    }
    with open(f"{ARTIFACTS_DIR}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"n = {n_rows} (Vagem/Acido, mesma fonte do pipeline de ML)")
    for target in ["qe_mgg", "remocao_pct"]:
        r = results[target]
        print(f"{target}: R²={r['r2']:.4f}  AdjR²={r['r2_adj']:.4f}  "
              f"cond.no={r['condition_number']:.1f}  F={r['model_f']:.2f} p={r['model_p']:.2e}")
    print("Otimo (desirability):", json.dumps(optimum, indent=2))
    print(f"Artefatos salvos em: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
