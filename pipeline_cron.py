# pipeline_cron.py
"""
Script A: Pipeline Persiapan Data & Clustering (Human-in-the-Loop)
Mengacu pada arsitektur CRISP-DM Fase 1 & Fase 2 untuk Log Wazuh SOC INSTIKI.

Alur:
1. Pembacaan dinamis log JSON (tar.gz / folder).
2. Interactive Agent Filtering (Blacklist input).
3. Data Cleaning & Flattening.
4. Feature Engineering (hour_group) & Deduplikasi Temporal (10 menit).
5. K-Modes Clustering (5 fitur agnostik).
6. Ekspor profil klaster & template pemetaan label semantik manual.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from kmodes.kmodes import KModes

from config import (
    DATA_RAW_DIR,
    DATA_PROC_DIR,
    REPORTS_DIR,
    PARQUET_CLEANED,
    PARQUET_DEDUPED,
    PARQUET_CLUSTERED,
    CSV_DEDUPED,
    CSV_PROFILES,
    CSV_CLUSTER_MAPPING,
    CLUSTER_FEATURES,
    KMODES_INIT,
    KMODES_N_INIT,
    KMODES_RANDOM_STATE,
)
from src.transformers import (
    read_wazuh_logs,
    flatten_and_clean_logs,
    apply_feature_engineering,
    deduplicate,
)

# ── Logging setup ──────────────────────────────────────────────────────
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(REPORTS_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# K optimal default (dapat disesuaikan hasil eksperimen elbow & silhouette)
K_OPTIMAL = 3


def filter_blacklist_agents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menampilkan semua agen yang terdeteksi dan memfilter berdasarkan input blacklist.
    Mendukung pemilihan via nomor urut (1, 2) maupun nama agen (case-insensitive).
    """
    detected_agents = sorted(df["agent_name"].dropna().unique().tolist())
    
    if not detected_agents:
        log.warning("Tidak ada agent_name yang ditemukan di dalam data.")
        return df

    print("\n" + "=" * 55)
    print("INTERACTIVE AGENT FILTERING (BLACKLIST)")
    print("=" * 55)
    print(f"Daftar agen terdeteksi ({len(detected_agents)} agen):")
    
    # Mapping nomor urut dan nama huruf kecil
    agent_map_by_index = {}
    agent_map_by_lower = {}
    for idx, agent in enumerate(detected_agents, 1):
        agent_map_by_index[str(idx)] = agent
        agent_map_by_lower[agent.lower()] = agent
        print(f"  [{idx}] {agent}")
    print("-" * 55)
    
    user_input = input(
        "Masukkan NOMOR atau NAMA agen yang ingin di-BLACKLIST\n"
        "(Contoh: '1, 3' atau 'test-server, project-1' | Tekan ENTER jika tidak ada): "
    ).strip()
    
    if not user_input:
        log.info("Tidak ada agen yang di-blacklist.")
        return df

    # Parsing input pengguna
    selected_tokens = [token.strip() for token in user_input.split(",") if token.strip()]
    blacklist = set()

    for token in selected_tokens:
        # Pengecekan via nomor urut
        if token in agent_map_by_index:
            blacklist.add(agent_map_by_index[token])
        # Pengecekan via nama (case-insensitive)
        elif token.lower() in agent_map_by_lower:
            blacklist.add(agent_map_by_lower[token.lower()])
        else:
            log.warning(f"Input '{token}' tidak cocok dengan agen mana pun, diabaikan.")

    if blacklist:
        log.info(f"Agen final yang masuk daftar blacklist: {list(blacklist)}")
        initial_count = len(df)
        df_filtered = df[~df["agent_name"].isin(blacklist)].copy()
        removed_count = initial_count - len(df_filtered)
        log.info(f"Data tersisa: {len(df_filtered):,} baris (Dibuang: {removed_count:,} baris)")
        return df_filtered

    log.info("Tidak ada agen valid yang dipilih untuk dibuang.")
    return df


def generate_cluster_profiles(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """Mengekstrak profil ringkas per cluster untuk analisis pakar."""
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
            "hour_group_dominan": subset["hour_group"].mode()[0] if not subset["hour_group"].empty else "-",
        })
    return pd.DataFrame(profiles)


def export_mapping_template(df_clustered: pd.DataFrame):
    """Menyiapkan template cluster_mapping.csv untuk pelabelan manual (Human-in-the-Loop)."""
    unique_clusters = sorted(df_clustered["cluster"].unique())
    mapping_data = []
    
    for cid in unique_clusters:
        subset = df_clustered[df_clustered["cluster"] == cid]
        sample_desc = subset["rule_description"].mode()[0] if "rule_description" in subset.columns and not subset["rule_description"].empty else "-"
        sample_rule = subset["rule_id"].mode()[0] if not subset["rule_id"].empty else "-"
        sample_level = subset["rule_level"].mode()[0] if not subset["rule_level"].empty else "-"
        
        mapping_data.append({
            "cluster": cid,
            "sample_rule_id": sample_rule,
            "sample_rule_level": sample_level,
            "sample_description": sample_desc,
            "label_semantik": ""  # Diisi manual oleh analis: misal System_Activity, Web_Attack
        })
        
    df_template = pd.DataFrame(mapping_data)
    df_template.to_csv(CSV_CLUSTER_MAPPING, index=False)
    log.info(f"Template mapping diekspor: {CSV_CLUSTER_MAPPING}")


def run_pipeline():
    start = datetime.now()
    log.info("=" * 60)
    log.info("Fase 1 & Fase 2: Pipeline Ekstraksi, Clustering & Profiling Dimulai")
    log.info(f"Waktu Mulai: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Input Dinamis & Ekstraksi Log ───────────────────────────
    log.info("Step 1: Membaca log Wazuh secara dinamis dari direktori...")
    raw_records = read_wazuh_logs(DATA_RAW_DIR)
    log.info(f"Total log mentah terbaca: {len(raw_records):,} baris")

    if not raw_records:
        log.error("Tidak ada data log yang berhasil dimuat. Proses dihentikan.")
        return

    # ── Step 2: Data Flattening & Interactive Blacklisting ─────────────
    log.info("Step 2: Meratakan struktur nested JSON & pembersihan awal...")
    df = flatten_and_clean_logs(raw_records)
    
    # Filter agen interaktif
    df = filter_blacklist_agents(df)

    df.to_parquet(PARQUET_CLEANED, index=False)
    log.info(f"Data cleaned disimpan di: {PARQUET_CLEANED}")

    # ── Step 3: Feature Engineering & Deduplikasi Temporal ─────────────
    log.info("Step 3: Feature engineering temporal & deduplikasi 10 menit...")
    df = apply_feature_engineering(df)
    deduped = deduplicate(df)

    log.info(f"Sebelum deduplikasi : {len(df):,} baris")
    log.info(f"Setelah deduplikasi : {len(deduped):,} pola unik")
    reduction_rate = (1 - (len(deduped) / len(df))) * 100
    log.info(f"Tingkat Reduksi Data: {reduction_rate:.2f}%")

    deduped.to_parquet(PARQUET_DEDUPED, index=False)
    deduped.to_csv(CSV_DEDUPED, index=False)
    log.info(f"Data hasil deduplikasi disimpan di: {PARQUET_DEDUPED}")

    # ── Step 4: K-Modes Clustering (Agnostik Infrastruktur) ─────────────
    log.info(f"Step 4: Menjalankan K-Modes Clustering (K={K_OPTIMAL}) pada fitur: {CLUSTER_FEATURES}...")
    X = deduped[CLUSTER_FEATURES].astype(str).values

    km = KModes(
        n_clusters=K_OPTIMAL,
        init=KMODES_INIT,
        n_init=KMODES_N_INIT,
        random_state=KMODES_RANDOM_STATE,
    )
    deduped["cluster"] = km.fit_predict(X)

    # Distribusi klaster
    dist = deduped["cluster"].value_counts().sort_index()
    for cid, count in dist.items():
        log.info(f"  Cluster {cid}: {count:,} pola unik")

    deduped.to_parquet(PARQUET_CLUSTERED, index=False)
    log.info(f"Data berklaster disimpan di: {PARQUET_CLUSTERED}")

    # ── Step 5: Ekspor Profil Klaster & Jeda Human-in-the-Loop ──────────
    log.info("Step 5: Mengekspor profil klaster dan template pemetaan semantik...")
    df_profiles = generate_cluster_profiles(deduped)
    df_profiles.to_csv(CSV_PROFILES, index=False)
    log.info(f"Profil klaster disimpan di: {CSV_PROFILES}")

    export_mapping_template(deduped)

    duration = (datetime.now() - start).seconds
    log.info("=" * 60)
    log.info(f"Script A selesai dalam {duration} detik.")
    print("\n" + "#" * 60)
    print("PROSES SELESAI: JEDA UNTUK HUMAN-IN-THE-LOOP")
    print(f"1. Buka file '{CSV_PROFILES}' untuk melihat ringkasan tiap klaster.")
    print(f"2. Buka file '{CSV_CLUSTER_MAPPING}' dan isi kolom 'label_semantik'")
    print("   (Contoh label: System_Activity, Web_Attack, Authentication, dll.)")
    print("3. Setelah selesai mengisi mapping, lanjutkan ke Script B (Pelatihan AI).")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline()