import os
import sqlite3

import joblib
import numpy as np
from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DB_PATH = os.path.join(STATIC_DIR, "db", "biosorcao.db")

FEATURES = ["ph", "massa_g", "tempo_min", "conc_inicial_mgL", "temperatura_C"]

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
model_qe = joblib.load(os.path.join(ARTIFACTS_DIR, "model_qe_mgg.joblib"))
model_remocao = joblib.load(os.path.join(ARTIFACTS_DIR, "model_remocao_pct.joblib"))


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204
    payload = request.get_json(force=True) or {}
    try:
        x = np.array([[float(payload[f]) for f in FEATURES]])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"payload invalido: {e}"}), 400

    x_s = scaler.transform(x)
    qe_pred = float(model_qe.predict(x_s)[0])
    remocao_pred = float(model_remocao.predict(x_s)[0])
    return jsonify({"qe_pred": qe_pred, "remocao_pred": remocao_pred})


def _rows_as_dicts(cur):
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


@app.route("/api/isotherms")
def isotherms():
    conn = sqlite3.connect(DB_PATH)
    try:
        params = _rows_as_dicts(conn.execute("""
            SELECT m.name AS material, t.name AS treatment, ip.temperature_C AS temperature_c,
                   ip.model, ip.qmax, ip.kl, ip.rl, ip.r2, ip.params_json, ip.notes, ip.sheet_source
            FROM isotherm_params ip
            JOIN treatments t ON ip.treatment_id = t.id
            JOIN materials m ON t.material_id = m.id
            ORDER BY m.name, t.name, ip.temperature_C, ip.model
        """))
        raw_points = _rows_as_dicts(conn.execute("""
            SELECT m.name AS material, t.name AS treatment, rp.temperature_C AS temperature_c,
                   rp.ci_mgL AS ci, rp.ce_mgL AS ce, rp.qe_mgg AS qe
            FROM isotherm_raw_points rp
            JOIN treatments t ON rp.treatment_id = t.id
            JOIN materials m ON t.material_id = m.id
            ORDER BY m.name, t.name, rp.temperature_C, rp.ce_mgL
        """))
    finally:
        conn.close()
    return jsonify({"params": params, "raw_points": raw_points})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8765, debug=False)
