# pipeline_c_evaluate.py
"""Script C: evaluasi model pada periode uji (di luar periode latih)."""
import json
import logging

import matplotlib
matplotlib.use("Agg")  # agar plt.show() tidak memblokir script

import joblib
import numpy as np
import pandas as pd

from config import (
    DATA_RAW_DIR,
    REPORTS_DIR,
    CSV_CLUSTER_MAPPING,
    CONFIG_RUN_JSON,
    MODEL_KMODES_FILE,
    CLUSTER_FEATURES,
    TARGET_LABEL_COL,
)
from src.transformers import (
    read_wazuh_logs,
    flatten_and_clean_logs,
    apply_feature_engineering,
    deduplicate,
    filter_period,
    period_bounds,
)
from src.classification import load_artifacts, encode_features, encode_target, evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def prompt_test_period():
    print("\nFormat: 2026 | 2026-03 | 2026-03-15")
    while True:
        start = input("Periode UJI - awal (wajib)   : ").strip()
        end = input("Periode UJI - akhir (ENTER = sama dengan awal): ").strip() or None
        if not start:
            print("Periode awal wajib diisi.")
            continue
        try:
            period_bounds(start)
            if end:
                period_bounds(end)
        except ValueError:
            print("Format tidak valid, coba lagi.")
            continue
        return start, end


def run():
    with open(CONFIG_RUN_JSON, encoding="utf-8") as f:
        run_cfg = json.load(f)
    blacklist = run_cfg.get("agen_diblacklist", [])
    train_p = run_cfg.get("periode_latih", {})
    log.info(f"Blacklist dari run latih: {blacklist}")
    log.info(f"Periode latih: {train_p}")

    model, encoders = load_artifacts()
    km = joblib.load(MODEL_KMODES_FILE)

    mapping = pd.read_csv(CSV_CLUSTER_MAPPING)
    label_map = dict(zip(mapping["cluster"].astype(int),
                         mapping["label_semantik"].astype(str).str.strip()))

    start, end = prompt_test_period()

    # Peringatan jika periode uji beririsan dengan periode latih
    if train_p.get("awal"):
        t_lo = period_bounds(train_p["awal"])[0]
        t_hi = period_bounds(train_p.get("akhir") or train_p["awal"])[1]
        s_lo = period_bounds(start)[0]
        s_hi = period_bounds(end or start)[1]
        if s_lo < t_hi and t_lo < s_hi:
            log.warning("Periode uji BERIRISAN dengan periode latih, hasil bisa terlalu optimis.")

    records = read_wazuh_logs(DATA_RAW_DIR)
    df = flatten_and_clean_logs(records)
    df = filter_period(df, start, end)
    df = df[~df["agent_name"].isin(blacklist)]
    if df.empty:
        log.error("Tidak ada data uji setelah filter periode & blacklist.")
        return
    log.info(f"Data uji: {len(df):,} log mentah")

    deduped = deduplicate(apply_feature_engineering(df))
    log.info(f"Pola uji setelah dedup: {len(deduped):,}")

    # Label uji: cluster dari K-Modes terlatih -> label semantik dari mapping
    X_str = deduped[CLUSTER_FEATURES].astype(str).values
    deduped["cluster"] = km.predict(X_str)
    deduped[TARGET_LABEL_COL] = deduped["cluster"].map(label_map)

    X = encode_features(deduped, encoders)
    y = encode_target(deduped, encoders)

    tag = f"{start}_{end or start}"
    res = evaluate_model(model, X, y, encoders, fig_name=f"confusion_matrix_uji_{tag}.png")

    unk = (X == -1)
    unknown = {
        "baris_dengan_nilai_asing_%": round(unk.any(axis=1).mean() * 100, 2),
        "berbobot_alert_count_%": round(
            np.average(unk.any(axis=1), weights=deduped["alert_count"]) * 100, 2),
        "per_fitur_%": {c: round(unk[:, i].mean() * 100, 2)
                        for i, c in enumerate(CLUSTER_FEATURES)},
    }
    print("\nUnknown rate (nilai tidak dikenal saat inferensi):")
    print(json.dumps(unknown, indent=2))

    out = {
        "periode_uji": {"awal": start, "akhir": end or start},
        "periode_latih": train_p,
        "log_mentah": len(df),
        "pola_dedup": len(deduped),
        "distribusi_label": deduped[TARGET_LABEL_COL].value_counts().to_dict(),
        "accuracy": res["accuracy"],
        "fpr": res["fpr"],
        "unknown_rate": unknown,
        "sumber_label_uji": "km.predict + cluster_mapping.csv",
    }
    path = REPORTS_DIR / f"eval_{tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Hasil evaluasi disimpan di: {path}")
    print("\nCatatan: label uji berasal dari K-Modes terlatih, sehingga akurasi mengukur "
          "seberapa baik Random Forest meniru K-Modes pada data baru.")


if __name__ == "__main__":
    run()