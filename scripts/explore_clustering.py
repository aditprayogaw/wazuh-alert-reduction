# scripts/explore_clustering.py
"""
Script 2: Eksplorasi Struktur Alert Tanpa Supervisi (K-Modes Clustering)
Khusus untuk analisis eksploratif dan visualisasi pada Bab 4 Laporan Skripsi.

Tujuan:
1. Menemukan K optimal (Elbow Method & Silhouette Score dengan Hamming Distance).
2. Membentuk klaster alami berdasarkan pola atribut kontekstual (rule_level, decoder_name, location, rule_groups).
3. Mengekstrak profil ringkas per klaster untuk menjelaskan struktur alert SOC INSTIKI.
4. Menyimpan artefak laporan ke reports/cluster_profiles.csv dan grafik ke reports/figures/.

Catatan:
- Skrip ini murni bersifat eksploratif (Unsupervised Exploratory Data Analysis).
- Tidak lagi digunakan untuk penentuan label otomatis (menghindari label leakage).
"""
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from kmodes.kmodes import KModes
import joblib

from config import (
    PARQUET_DEDUPED,
    CLUSTER_FEATURES,
    K_RANGE,
    K_SEARCH_SAMPLE,
    KMODES_INIT,
    KMODES_N_INIT,
    KMODES_RANDOM_STATE,
    REPORTS_DIR,
    FIGURES_DIR,
    CSV_PROFILES,
    CSV_K_SELECTION,
    MODEL_KMODES_FILE,
)
from src.clustering import (
    compute_elbow,
    compute_silhouette,
    plot_elbow,
    plot_silhouette,
)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def generate_cluster_profiles(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """Mengekstrak profil ringkas per klaster untuk analisis karakteristik data di laporan."""
    profiles = []
    for cid in sorted(df_clustered["cluster"].unique()):
        subset = df_clustered[df_clustered["cluster"] == cid]
        profiles.append({
            "cluster": cid,
            "total_pola_dedup": len(subset),
            "total_alert_raw": subset["alert_count"].sum() if "alert_count" in subset.columns else len(subset),
            "rule_id_dominan": subset["rule_id"].mode()[0] if not subset["rule_id"].empty else "-",
            "rule_level_dominan": subset["rule_level"].mode()[0] if not subset["rule_level"].empty else "-",
            "decoder_dominan": subset["decoder_name"].mode()[0] if not subset["decoder_name"].empty else "-",
            "location_dominan": subset["location"].mode()[0] if not subset["location"].empty else "-",
            "rule_groups_dominan": subset["rule_groups"].mode()[0] if not subset["rule_groups"].empty else "-",
        })
    return pd.DataFrame(profiles)


def run_pipeline():
    log.info("=" * 65)
    log.info("SKRIP 2: Eksplorasi Struktur Alert Tanpa Supervisi (K-Modes)")
    log.info("=" * 65)

    if not PARQUET_DEDUPED.exists():
        log.error(f"File {PARQUET_DEDUPED} tidak ditemukan. Jalankan scripts/prepare_data.py terlebih dahulu.")
        return

    # 1. Baca Data Terdeduplikasi
    df = pd.read_parquet(PARQUET_DEDUPED)
    log.info(f"Data terbaca: {len(df):,} pola terdeduplikasi.")
    log.info(f"Fitur clustering konteks: {CLUSTER_FEATURES}")

    # 2. Pencarian K Optimal (Elbow & Silhouette)
    X_full = df[CLUSTER_FEATURES].astype(str).values
    sample_size = min(K_SEARCH_SAMPLE, len(X_full))
    log.info(f"Mengambil sampel {sample_size:,} baris untuk pencarian K optimal...")
    
    # Ambil sampel acak yang representatif
    np_state = np.random.RandomState(KMODES_RANDOM_STATE)
    sample_indices = np_state.choice(len(X_full), size=sample_size, replace=False)
    X_sample = X_full[sample_indices]

    log.info("Menghitung Cost Elbow...")
    costs = compute_elbow(X_sample)
    plot_elbow(costs)

    log.info("Menghitung Silhouette Score (Metrik Hamming)...")
    sil_scores = compute_silhouette(X_sample)
    plot_silhouette(sil_scores)

    # Simpan hasil pencarian K
    df_k = pd.DataFrame({"k": list(K_RANGE), "cost": costs, "silhouette": sil_scores})
    df_k.to_csv(CSV_K_SELECTION, index=False)
    log.info(f"Hasil seleksi K disimpan di: {CSV_K_SELECTION}")

    # K yang disarankan (silhouette tertinggi)
    valid_sil = [(k, s) for k, s in zip(K_RANGE, sil_scores) if s is not None]
    k_suggested = max(valid_sil, key=lambda x: x[1])[0] if valid_sil else 3
    print(f"\n* Rekomendasi K berdasarkan Silhouette Score tertinggi: K = {k_suggested}")

    user_k = input(f"Masukkan jumlah K yang diinginkan (ENTER = pakai K={k_suggested}): ").strip()
    k_final = int(user_k) if user_k.isdigit() else k_suggested

    # 3. Fitting K-Modes Final
    log.info(f"Menjalankan K-Modes Clustering Final dengan K = {k_final}...")
    km = KModes(
        n_clusters=k_final,
        init=KMODES_INIT,
        n_init=KMODES_N_INIT,
        random_state=KMODES_RANDOM_STATE,
    )
    df["cluster"] = km.fit_predict(X_full)

    # 4. Ekspor Profil Klaster
    profiles = generate_cluster_profiles(df)
    profiles.to_csv(CSV_PROFILES, index=False)
    log.info(f"Profil klaster disimpan di: {CSV_PROFILES}")

    # Simpan model
    joblib.dump(km, MODEL_KMODES_FILE)
    log.info(f"Model K-Modes disimpan di: {MODEL_KMODES_FILE}")

    print("\n" + "=" * 65)
    print("PROFIL KLASTER ALERT TERSUSUN:")
    print("=" * 65)
    print(profiles.to_string(index=False))
    print("=" * 65)
    print(f"Grafik Elbow & Silhouette tersimpan di folder: {FIGURES_DIR}")
    print("Skrip 2 selesai!\n")


if __name__ == "__main__":
    import numpy as np
    run_pipeline()

