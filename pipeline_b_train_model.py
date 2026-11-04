# pipeline_b_train_model.py
"""
Script B: Pelatihan Model Klasifikasi (Random Forest)
Mengacu pada arsitektur CRISP-DM Fase 4 (Modeling: Classification) untuk Log Wazuh SOC INSTIKI.

Alur:
1. Baca data berklaster (clustered.parquet) hasil Script A.
2. Beri label tiap baris berdasarkan rule_id (data/processed/rule_labels.csv, kolom label_final).
3. Stratified split 80/20, Custom Label Encoder (nilai asing -> -1).
4. Latih Random Forest (class_weight balanced untuk kelas timpang).
5. Evaluasi: accuracy, classification report, confusion matrix, FPR per kelas.
6. Simpan model dan encoder ke models/.

Keluaran utama:
- models/random_forest_wazuh.pkl, models/custom_label_encoders.pkl
- reports/eval_latih.json, reports/figures/confusion_matrix_rf.png

Catatan: split acak pada periode yang sama membuat rule_id uji hampir pasti sudah dikenal,
sehingga akurasinya cenderung tinggi. Evaluasi generalisasi ada di Script C.
"""
import json

import matplotlib
matplotlib.use("Agg")  # agar plt.show() tidak memblokir script

import pandas as pd

from config import PARQUET_CLUSTERED, REPORTS_DIR, TARGET_LABEL_COL, CSV_RULE_LABELS
from src.classification import (
    apply_rule_labels, split_data, fit_encoders, encode_features,
    encode_target, train_random_forest, evaluate_model, save_artifacts,
)


def run_pipeline():
    df = pd.read_parquet(PARQUET_CLUSTERED)
    df = apply_rule_labels(df)

    print(f"Label diambil dari: {CSV_RULE_LABELS}")
    print("Distribusi label (pola dedup):")
    print(df[TARGET_LABEL_COL].value_counts().to_string())
    print("\nDistribusi label (alert):")
    print(df.groupby(TARGET_LABEL_COL)["alert_count"].sum().sort_values(ascending=False).to_string())

    train_df, test_df = split_data(df)
    encoders = fit_encoders(train_df)

    X_train, y_train = encode_features(train_df, encoders), encode_target(train_df, encoders)
    X_test, y_test = encode_features(test_df, encoders), encode_target(test_df, encoders)

    model = train_random_forest(X_train, y_train)
    res = evaluate_model(model, X_test, y_test, encoders)
    save_artifacts(model, encoders)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "pola_latih": len(train_df),
        "pola_uji_internal": len(test_df),
        "distribusi_label_pola": df[TARGET_LABEL_COL].value_counts().to_dict(),
        "distribusi_label_alert": df.groupby(TARGET_LABEL_COL)["alert_count"].sum().to_dict(),
        "accuracy": res["accuracy"],
        "fpr": res["fpr"],
        "catatan": "Split acak 80/20 pada periode yang sama: rule_id uji hampir pasti sudah dikenal. "
                   "Evaluasi generalisasi ada di Script C.",
    }
    with open(REPORTS_DIR / "eval_latih.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print("\nScript B selesai. Model dan encoder tersimpan di folder models/.")


if __name__ == "__main__":
    run_pipeline()