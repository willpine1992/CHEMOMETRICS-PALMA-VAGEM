"""
Ajuste Nao Linear de Isotermas e Cinetica + Analise Estatistica de Erros.

Fonte: mesmo banco relacional (db/biosorcao.db) usado pelos demais paineis.
- Isotermas: tabela isotherm_raw_points (36 pontos Ci/Ce/Qe de Vagem, ja extraidos
  pelo ETL a partir da aba "Isotermas VAGEM" -- ver etl/build_database.py). Cada
  bloco (material, tratamento, temperatura) tem so 4 pontos -- poucos para modelos
  de 3 parametros (Redlich-Peterson), tratar os erros/IC com cautela nesse caso,
  documentado explicitamente nos artefatos e no dashboard.
- Cinetica: experiment_runs com experiment_type='tempo' (abas "Tempo (Vagem)",
  "Tempo (Vagem 2)", "Tempo PALMA") -- Qe medido ao longo do tempo de contato,
  duas replicas por timepoint (duas abas/corridas independentes para Vagem).
  Usamos a MEDIA das replicas em cada timepoint para o ajuste (pratica padrao em
  cinetica de adsorcao), mas guardamos os pontos brutos tambem para plotar como
  simbolos de fundo no grafico.

Todos os ajustes sao NAO LINEARES diretos (scipy.optimize.curve_fit, que usa
Levenberg-Marquardt por padrao quando nao ha bounds, e Trust Region Reflective
quando ha) sobre a equacao original de cada modelo -- sem nenhuma linearizacao
algebrica (ver a analise anterior do bloco Vagem/Base/15C, onde a linearizacao deu
R2=0.03 enquanto o ajuste direto deu R2=0.87 nos MESMOS dados: linearizar distorce
a estrutura do erro).
"""
import json
import os
import sqlite3

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DB_PATH = "/Users/macbookpro/Documents/POSDOC - MAC/QUIMIOINFORMÁTICA/DASHBOARD-VAGEM/db/biosorcao.db"
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

R_GAS = 8.314  # J/(mol*K)


# ---------------------------------------------------------------- modelos ----

def langmuir(ce, qmax, kl):
    return (qmax * kl * ce) / (1.0 + kl * ce)


def freundlich(ce, kf, n):
    return kf * np.power(ce, 1.0 / n)


def make_temkin(temp_k):
    def temkin(ce, at, bt):
        ce_safe = np.clip(ce, 1e-9, None)
        return (R_GAS * temp_k / bt) * np.log(at * ce_safe)
    return temkin


def redlich_peterson(ce, k, a, g):
    return (k * ce) / (1.0 + a * np.power(ce, g))


def pfo(t, qe, k1):
    return qe * (1.0 - np.exp(-k1 * t))


def pso(t, qe, k2):
    denom = 1.0 + k2 * qe * t
    return (k2 * (qe ** 2) * t) / denom


def elovich(t, alpha, beta):
    return (1.0 / beta) * np.log(1.0 + alpha * beta * t)


ISOTHERM_MODELS = {
    "Langmuir": dict(func=langmuir, n_params=2,
                      p0=lambda ce, qe: [max(qe) * 1.2, 0.01],
                      bounds=([1e-6, 1e-6], [np.inf, np.inf]),
                      param_names=["qmax", "kl"]),
    "Freundlich": dict(func=freundlich, n_params=2,
                        p0=lambda ce, qe: [max(qe.mean(), 0.1), 2.0],
                        bounds=([1e-6, 0.05], [np.inf, 20.0]),
                        param_names=["kf", "n"]),
    "Redlich-Peterson": dict(func=redlich_peterson, n_params=3,
                              p0=lambda ce, qe: [max(qe) * 0.05, 0.01, 1.0],
                              bounds=([1e-8, 1e-8, 0.0], [np.inf, np.inf, 2.0]),
                              param_names=["k", "a", "g"]),
}

KINETIC_MODELS = {
    "PFO": dict(func=pfo, n_params=2,
                p0=lambda t, qt: [max(qt) * 1.1, 0.01],
                bounds=([1e-6, 1e-6], [np.inf, np.inf]),
                param_names=["qe", "k1"]),
    "PSO": dict(func=pso, n_params=2,
                p0=lambda t, qt: [max(qt) * 1.1, 0.01],
                bounds=([1e-6, 1e-8], [np.inf, np.inf]),
                param_names=["qe", "k2"]),
    "Elovich": dict(func=elovich, n_params=2,
                     p0=lambda t, qt: [1.0, 0.5],
                     bounds=([1e-8, 1e-8], [np.inf, np.inf]),
                     param_names=["alpha", "beta"]),
}


# ---------------------------------------------------------------- erros ----

def error_functions(y_exp, y_calc, n_params):
    y_exp = np.asarray(y_exp, dtype=float)
    y_calc = np.asarray(y_calc, dtype=float)
    n = len(y_exp)

    ss_res = float(np.sum((y_exp - y_calc) ** 2))
    ss_tot = float(np.sum((y_exp - y_exp.mean()) ** 2))
    r2_nl = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    # chi2/ARE/HYBRID dividem por y_exp ou y_calc -- exclui pontos ~0 (comuns em
    # t=0 nos dados de cinetica) para nao gerar divisao por zero / valores absurdos.
    mask = (np.abs(y_exp) > 1e-6) & (np.abs(y_calc) > 1e-6)
    n_valid = int(mask.sum())
    if n_valid == 0:
        chi2 = are = hybrid = None
    else:
        ye, yc = y_exp[mask], y_calc[mask]
        chi2 = float(np.sum((ye - yc) ** 2 / yc))
        are = float(100.0 / n_valid * np.sum(np.abs(ye - yc) / np.abs(ye)))
        dof = max(n_valid - n_params, 1)
        hybrid = float(100.0 / dof * np.sum((ye - yc) ** 2 / np.abs(ye)))

    return dict(chi2=chi2, are=are, hybrid=hybrid, r2_nl=r2_nl,
                n_points=n, n_points_used_for_relative_errors=n_valid)


def fit_one(model_key, model_spec, x, y, extra_func=None):
    func = extra_func or model_spec["func"]
    try:
        p0 = model_spec["p0"](x, y)
        popt, pcov = curve_fit(func, x, y, p0=p0, bounds=model_spec["bounds"], maxfev=20000)
        y_calc = func(x, *popt)
        errors = error_functions(y, y_calc, model_spec["n_params"])
        perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else [None] * len(popt)
        params = {name: float(val) for name, val in zip(model_spec["param_names"], popt)}
        param_stderr = {name: (float(e) if e is not None else None)
                         for name, e in zip(model_spec["param_names"], perr)}
        return dict(ok=True, params=params, param_stderr=param_stderr, errors=errors,
                    residuals=(np.asarray(y) - y_calc).tolist())
    except Exception as e:
        return dict(ok=False, error=str(e))


def curve_points(func, x_lo, x_hi, extra_params, n=60):
    xs = np.linspace(max(x_lo, 0), x_hi, n)
    ys = func(xs, *extra_params)
    return xs.tolist(), ys.tolist()


# ---------------------------------------------------------------- dados ----

def load_isotherm_groups():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("""
            SELECT m.name AS material, t.name AS treatment, rp.temperature_C AS temperature_c,
                   rp.ci_mgL AS ci, rp.ce_mgL AS ce, rp.qe_mgg AS qe
            FROM isotherm_raw_points rp
            JOIN treatments t ON rp.treatment_id = t.id
            JOIN materials m ON t.material_id = m.id
            ORDER BY m.name, t.name, rp.temperature_C, rp.ce_mgL
        """, conn)
    finally:
        conn.close()
    return df


def load_kinetics_groups():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("""
            SELECT m.name AS material, t.name AS treatment, er.tempo_min AS t_min, er.qe_mgg AS qe
            FROM experiment_runs er
            JOIN experiments e ON er.experiment_id = e.id
            JOIN treatments t ON e.treatment_id = t.id
            JOIN materials m ON t.material_id = m.id
            WHERE e.experiment_type = 'tempo'
            ORDER BY m.name, t.name, er.tempo_min
        """, conn)
    finally:
        conn.close()
    return df


# ---------------------------------------------------------------- pipeline ----

def process_isotherms():
    df = load_isotherm_groups()
    groups = []
    for (material, treatment, temp_c), g in df.groupby(["material", "treatment", "temperature_c"]):
        ce = g["ce"].values.astype(float)
        qe = g["qe"].values.astype(float)
        temp_k = temp_c + 273.15

        # Dominio de plotagem das curvas: comeca perto do menor Ce observado (nao em
        # Ce=0) porque Temkin diverge para -infinito quando Ce->0 (ln(AT*Ce)) -- plotar
        # dessa regiao nao-fisica/nao-observada distorceria a escala do grafico inteiro
        # (mesmo dominio usado p/ todos os modelos, para ficarem comparaveis).
        x_lo = ce.min() * 0.5
        x_hi = ce.max() * 1.15

        models_out = {}
        for name, spec in ISOTHERM_MODELS.items():
            res = fit_one(name, spec, ce, qe)
            if res["ok"]:
                xs, ys = curve_points(spec["func"], x_lo, x_hi, list(res["params"].values()))
                res["curve"] = {"ce": xs, "qe": ys}
                res["residuals_x"] = ce.tolist()
            models_out[name] = res

        # Temkin precisa da temperatura (K) como parametro FIXO, nao ajustado -- tratado
        # a parte pois sua assinatura de funcao depende do grupo (temperatura do bloco).
        temkin_func = make_temkin(temp_k)
        temkin_spec = dict(func=temkin_func, n_params=2,
                            p0=lambda ce_, qe_: [1.0, 500.0],
                            bounds=([1e-6, 1.0], [np.inf, np.inf]),
                            param_names=["AT", "bT"])
        res = fit_one("Temkin", temkin_spec, ce, qe, extra_func=temkin_func)
        if res["ok"]:
            xs, ys = curve_points(temkin_func, x_lo, x_hi, list(res["params"].values()))
            res["curve"] = {"ce": xs, "qe": ys}
            res["residuals_x"] = ce.tolist()
        models_out["Temkin"] = res

        groups.append(dict(
            material=material, treatment=treatment, temperature_c=float(temp_c),
            n_points=len(ce), data={"ce": ce.tolist(), "qe": qe.tolist()},
            models=models_out,
        ))
    return {"groups": groups}


def process_kinetics():
    df = load_kinetics_groups()
    groups = []
    for (material, treatment), g in df.groupby(["material", "treatment"]):
        g_mean = g.groupby("t_min", as_index=False)["qe"].mean().sort_values("t_min")
        t = g_mean["t_min"].values.astype(float)
        qt = g_mean["qe"].values.astype(float)

        models_out = {}
        for name, spec in KINETIC_MODELS.items():
            res = fit_one(name, spec, t, qt)
            if res["ok"]:
                xs, ys = curve_points(spec["func"], 0, t.max() * 1.05, list(res["params"].values()))
                res["curve"] = {"t": xs, "qt": ys}
                res["residuals_x"] = t.tolist()
            models_out[name] = res

        groups.append(dict(
            material=material, treatment=treatment, n_points=len(t),
            data={"t": t.tolist(), "qt": qt.tolist()},
            data_raw={"t": g["t_min"].tolist(), "qt": g["qe"].tolist()},
            models=models_out,
        ))
    return {"groups": groups}


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    iso = process_isotherms()
    with open(f"{ARTIFACTS_DIR}/isotherm_fits.json", "w") as fh:
        json.dump(iso, fh)
    print(f"Isotermas: {len(iso['groups'])} grupos (material x tratamento x temperatura)")
    for grp in iso["groups"][:3]:
        line = f"  {grp['material']}/{grp['treatment']}/{grp['temperature_c']}C (n={grp['n_points']}): "
        line += ", ".join(
            f"{name}=R2NL:{m['errors']['r2_nl']:.3f}" if m.get("ok") and m["errors"]["r2_nl"] is not None else f"{name}=FALHOU"
            for name, m in grp["models"].items()
        )
        print(line)

    kin = process_kinetics()
    with open(f"{ARTIFACTS_DIR}/kinetics_fits.json", "w") as fh:
        json.dump(kin, fh)
    print(f"\nCinética: {len(kin['groups'])} grupos (material x tratamento)")
    for grp in kin["groups"]:
        line = f"  {grp['material']}/{grp['treatment']} (n={grp['n_points']}): "
        line += ", ".join(
            f"{name}=R2NL:{m['errors']['r2_nl']:.3f}" if m.get("ok") and m["errors"]["r2_nl"] is not None else f"{name}=FALHOU"
            for name, m in grp["models"].items()
        )
        print(line)

    print(f"\nArtefatos salvos em: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
