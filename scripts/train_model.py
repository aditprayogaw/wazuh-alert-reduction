# scripts/train_model.py
"""
Script 3: Pelatihan Model Klasifikasi Random Forest Berbasis Tabel Kebijakan Rule

Fungsi Utama:
1. Memuat data terdeduplikasi (data/processed/deduped.parquet).
2. Menghubungkan log dengan tabel kebijakan resmi (data/processed/rule_labels_done.csv).
3. Melatih CustomLabelEncoder pada fitur konteks:
   - rule_level (numerik)
   - decoder_name (kategorikal)
   - location (kategorikal)
   - rule_groups (kategorikal)
4. Melatih Random Forest Classifier pada seluruh populasi rule dikenal.
5. Menyimpan artefak model ke models/random_forest_wazuh.pkl dan encoder ke models/custom_label_encoders.pkl.
"""
import logging
from pathlib import Path
import pandas as pd

from config import (
    PARQUET_DEDUPED,
    CSV_RULE_LABELS,
    TARGET_LABEL_COL,
    CLASSIFICATION_FEATURES,
    MODEL_RF_FILE,
    ENCODER_FILE,
)
from src.classification import (
    apply_rule_labels,
    fit_encoders,
    encode_features,
    encode_target,
    train_random_forest,
    save_artifacts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_pipeline():
    log.info("=" * 65)
    log.info("SKRIP 3: Pelatihan Model Random Forest Dimulai")
    log.info("=" * 65)

    if not PARQUET_DEDUPED.exists():
        log.error(f"File {PARQUET_DEDUPED} belum ada. Jalankan scripts/prepare_data.py terlebih dahulu.")
        return

    if not CSV_RULE_LABELS.exists():
        log.error(f"Tabel kebijakan {CSV_RULE_LABELS} tidak ditemukan.")
        return

    # 1. Muat Data & Pasangkan Label Kebijakan
    log.info("Membaca data terdeduplikasi & memasangkan label dari tabel kebijakan...")
    df = pd.read_parquet(PARQUET_DEDUPED)
    df = apply_rule_labels(df)

    log.info(f"Tabel kebijakan terbaca: {CSV_RULE_LABELS}")
    log.info(f"Total baris pola terdeduplikasi: {len(df):,}")

    # Distribusi Label
    print("\n" + "-" * 50)
    print("DISTRIBUSI LABEL TRIASE PADA DATA:")
    print("-" * 50)
    pola_dist = df[TARGET_LABEL_COL].value_counts()
    alert_dist = df.groupby(TARGET_LABEL_COL)["alert_count"].sum()
    for lbl in pola_dist.index:
        p_cnt = pola_dist[lbl]
        a_cnt = alert_dist[lbl]
        print(f"  * {lbl:<10} : {p_cnt:>6,} pola dedup ({p_cnt/len(df)*100:>5.1f}%) | {a_cnt:>8,} total alert ({a_cnt/alert_dist.sum()*100:>5.1f}%)")
    print("-" * 50 + "\n")

    # 2. Fit Encoders
    log.info(f"Menyiapkan encoder fitur konteks: {CLASSIFICATION_FEATURES}")
    encoders = fit_encoders(df)

    # 3. Transformasi Fitur
    X = encode_features(df, encoders)
    y = encode_target(df, encoders)

    # 4. Melatih Random Forest
    log.info("Melatih Random Forest Classifier (Class Weight: Balanced)...")
    model = train_random_forest(X, y)

    # 5. Analisis Kontribusi Fitur (Feature Importance)
    log.info("Menganalisis Feature Importance:")
    for col, imp in sorted(zip(CLASSIFICATION_FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True):
        log.info(f"  * {col:<15} : {imp*100:5.2f}%")

    # 6. Simpan Artefak
    save_artifacts(model, encoders)
    log.info(f"Model tersimpan di   : {MODEL_RF_FILE}")
    log.info(f"Encoder tersimpan di : {ENCODER_FILE}")

    print("\n" + "#" * 65)
    print("PELATIHAN SELESAI!")
    print("Model produksi Random Forest dan encoder berhasil disimpan.")
    print("Langkah berikutnya: Jalankan evaluasi 5-Fold Cross-Validation Berbasis Rule:")
    print("  python -m scripts.evaluate_model")
    print("#" * 65 + "\n")


if __name__ == "__main__":
    run_pipeline()
