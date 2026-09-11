"""
Pipeline de Modelagem Preditiva ML + XAI para adsorcao da fibra de Vagem (tratamento Acido).

Etapas: winsorizacao de outliers -> StandardScaler -> train/test split 75/25 ->
GridSearchCV (RandomForest, XGBoost, SVR) com k-fold CV -> metricas R2/RMSE/MAE ->
SHAP no melhor modelo -> salva artefatos (joblib + json) para o dashboard/simulador.

Dataset pequeno (49 linhas) e real: metricas de teste podem ser modestas/instaveis -
isso e esperado e reportado com transparencia (ver metrics.json e relatorio da fase).
"""
import json
import sqlite3
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DB_PATH = "/Users/macbookpro/Documents/POSDOC - MAC/QUIMIOINFORMÁTICA/DASHBOARD-VAGEM/db/biosorcao.db"
ARTIFACTS_DIR = "/Users/macbookpro/Documents/POSDOC - MAC/QUIMIOINFORMÁTICA/DASHBOARD-VAGEM/ml/artifacts"

# Fonte de dados: banco relacional SQLite (populado por etl/build_database.py a partir do
# Google Drive), no lugar do CSV estatico anterior. Mesmo recorte de sempre -- material
# Vagem, tratamento Acido -- para manter os resultados de ML comparaveis ao longo do tempo.
SQL_QUERY = """
SELECT
  er.ph AS ph,
  er.massa_g AS massa_g,
  er.tempo_min AS tempo_min,
  er.conc_inicial_mgL AS conc_inicial_mgL,
  er.temperatura_C AS temperatura_C,
  er.qe_mgg AS qe_mgg,
  er.remocao_pct AS remocao_pct
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

FEATURES = ["ph", "massa_g", "tempo_min", "conc_inicial_mgL", "temperatura_C"]
FEATURE_LABELS = {
    "ph": "pH",
    "massa_g": "Massa (g)",
    "tempo_min": "Tempo (min)",
    "conc_inicial_mgL": "Concentração inicial (mg/L)",
    "temperatura_C": "Temperatura (°C)",
}
RANDOM_STATE = 42


def winsorize_iqr(df, cols, k=1.5):
    df = df.copy()
    report = {}
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_clipped = ((df[c] < lo) | (df[c] > hi)).sum()
        df[c] = df[c].clip(lo, hi)
        report[c] = {"lower": float(lo), "upper": float(hi), "n_clipped": int(n_clipped)}
    return df, report


def evaluate(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_target(df, target_col, n_rows):
    X = df[FEATURES].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grids = {
        "random_forest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 200, 400],
                "max_depth": [None, 4, 8],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "xgboost": (
            XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror"),
            {
                "n_estimators": [100, 200, 400],
                "max_depth": [2, 3, 5],
                "learning_rate": [0.03, 0.1, 0.2],
            },
        ),
        "svr": (
            SVR(),
            {
                "C": [1, 10, 100],
                "epsilon": [0.01, 0.1, 0.5],
                "kernel": ["rbf"],
                "gamma": ["scale", "auto"],
            },
        ),
    }

    results = {}
    fitted = {}
    for name, (estimator, grid) in grids.items():
        gs = GridSearchCV(estimator, grid, cv=cv, scoring="r2", n_jobs=-1)
        gs.fit(X_train_s, y_train)
        best = gs.best_estimator_
        train_pred = best.predict(X_train_s)
        test_pred = best.predict(X_test_s)
        results[name] = {
            "train": evaluate(y_train, train_pred),
            "test": evaluate(y_test, test_pred),
            "best_params": gs.best_params_,
        }
        fitted[name] = best

    best_model_name = max(
        results,
        key=lambda k: (results[k]["test"]["r2"], -results[k]["test"]["rmse"]),
    )
    best_model = fitted[best_model_name]

    metrics = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "models": results,
        "best_model": best_model_name,
    }

    parity = {
        "train": {"y_true": y_train.tolist(), "y_pred": best_model.predict(X_train_s).tolist()},
        "test": {"y_true": y_test.tolist(), "y_pred": best_model.predict(X_test_s).tolist()},
    }

    # SHAP no dataset completo (padronizado) para maximizar cobertura dado o N pequeno
    X_all_s = scaler.transform(X)
    if best_model_name in ("random_forest", "xgboost"):
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_all_s)
    else:
        background = shap.sample(X_train_s, min(20, len(X_train_s)), random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(best_model.predict, background)
        shap_values = explainer.shap_values(X_all_s, nsamples=100)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs_shap)
    feature_order = [FEATURE_LABELS[FEATURES[i]] for i in order]

    points = []
    for fi in range(len(FEATURES)):
        fname = FEATURE_LABELS[FEATURES[fi]]
        raw_vals = X[:, fi]
        vmin, vmax = raw_vals.min(), raw_vals.max()
        span = (vmax - vmin) or 1.0
        for si in range(len(raw_vals)):
            norm = (raw_vals[si] - vmin) / span
            points.append({
                "feature": fname,
                "feature_value": float(raw_vals[si]),
                "feature_value_norm": float(norm),
                "shap_value": float(shap_values[si, fi]),
            })

    shap_json = {"feature_order": feature_order, "points": points}

    # refit final do melhor modelo em TODOS os dados de treino (ja feito acima) - salva scaler+modelo
    joblib.dump(best_model, f"{ARTIFACTS_DIR}/model_{target_col}.joblib")

    return metrics, parity, shap_json, scaler, best_model_name, results


def main():
    df = load_dataset()
    n_rows = len(df)

    # Winsoriza apenas os ALVOS (Qe, %R), que sao medidas sujeitas a ruido experimental.
    # As FEATURES sao niveis de fator controlados pelo experimentador (setpoints), nao
    # medidas com erro aleatorio - "corrigir outliers" nelas nao faz sentido estatistico
    # e, num dataset pequeno com poucos niveis por fator (ex.: temperatura so tem 3
    # valores reais: 15/30/45C, com 25C assumido no restante), o IQR de uma feature pode
    # colapsar (IQR=0) e a winsorizacao acabaria apagando os unicos pontos que carregam
    # sinal real (ex.: os 12 pontos de 15/30/45C seriam clipados para 25C). Por isso
    # winsorizamos so qe_mgg e remocao_pct.
    df_clean, winsor_report = winsorize_iqr(df, ["qe_mgg", "remocao_pct"])

    qe_metrics, qe_parity, qe_shap, scaler, qe_best, qe_all = train_target(df_clean, "qe_mgg", n_rows)
    rem_metrics, rem_parity, rem_shap, _, rem_best, rem_all = train_target(df_clean, "remocao_pct", n_rows)

    joblib.dump(scaler, f"{ARTIFACTS_DIR}/scaler.joblib")

    metrics_all = {"qe": qe_metrics, "remocao": rem_metrics}
    with open(f"{ARTIFACTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics_all, f, indent=2)

    with open(f"{ARTIFACTS_DIR}/parity_qe.json", "w") as f:
        json.dump(qe_parity, f)
    with open(f"{ARTIFACTS_DIR}/parity_remocao.json", "w") as f:
        json.dump(rem_parity, f)

    with open(f"{ARTIFACTS_DIR}/shap_qe.json", "w") as f:
        json.dump(qe_shap, f)
    with open(f"{ARTIFACTS_DIR}/shap_remocao.json", "w") as f:
        json.dump(rem_shap, f)

    ranges = {}
    for feat in FEATURES:
        ranges[feat] = {
            "label": FEATURE_LABELS[feat],
            "min": float(df[feat].min()),
            "max": float(df[feat].max()),
            "default": float(df[feat].median()),
        }
    with open(f"{ARTIFACTS_DIR}/feature_ranges.json", "w") as f:
        json.dump(ranges, f, indent=2)

    report = {
        "n_rows_total": n_rows,
        "winsorization": winsor_report,
        "qe_best_model": qe_best,
        "remocao_best_model": rem_best,
    }
    with open(f"{ARTIFACTS_DIR}/run_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=== QE metrics ===")
    print(json.dumps(qe_metrics, indent=2))
    print("=== REMOCAO metrics ===")
    print(json.dumps(rem_metrics, indent=2))
    print("=== Winsorization ===")
    print(json.dumps(winsor_report, indent=2))
    print("Top features (Qe, por |SHAP| medio):", qe_shap["feature_order"])
    print("Top features (Remocao, por |SHAP| medio):", rem_shap["feature_order"])
    print("OK - artefatos salvos em", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
