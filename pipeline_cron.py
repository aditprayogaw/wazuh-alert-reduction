# pipeline_cron.py
"""
Script A: Pipeline Persiapan Data & Clustering (Human-in-the-Loop)
Mengacu pada arsitektur CRISP-DM Fase 3 (Data Preparation) & Fase 4 (Modeling:
Clustering) untuk Log Wazuh SOC INSTIKI.

Alur:
1. Tentukan sumber data (path arsip/folder log, disalin ke data/raw/).
2. Pembacaan dinamis log JSON (tar.gz / folder) & flattening + cleaning.
3. Pemilihan periode data latih (tahun / bulan / tanggal) secara interaktif.
4. Interactive Agent Filtering (blacklist agen non-target).
5. Feature Engineering (hour_group) & Deduplikasi Temporal (10 menit).
6. Pencarian K optimal (Elbow + Silhouette), K dipilih user.
7. K-Modes Clustering (5 fitur agnostik infrastruktur), model disimpan.
8. Ekspor profil klaster, template pemetaan label semantik manual,
   dan catatan konfigurasi run (reports/run_config.json).

Keluaran utama:
- data/processed/{cleaned,deduped,clustered}.parquet, deduped.csv
- data/processed/cluster_mapping.csv   (diisi manual oleh analis SOC)
- reports/cluster_profiles.csv, k_selection.csv, run_config.json, pipeline.log
- models/kmodes_wazuh.pkl

Langkah berikutnya: isi kolom label_semantik, lalu jalankan Script B.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # agar plt.show() tidak memblokir script

import joblib
import pandas as pd
from kmodes.kmodes import KModes

from config import (
    DATA_RAW_DIR,
    DATA_PROC_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    PARQUET_CLEANED,
    PARQUET_DEDUPED,
    PARQUET_CLUSTERED,
    CSV_DEDUPED,
    CSV_PROFILES,
    CSV_RULE_LABELS,
    CSV_K_SELECTION,
    CONFIG_RUN_JSON,
    MODEL_KMODES_FILE,
    CLUSTER_FEATURES,
    TIME_WINDOW_MINUTES,
    K_RANGE,
    K_SEARCH_SAMPLE,
    KMODES_INIT,
    KMODES_N_INIT,
    KMODES_RANDOM_STATE,
    CSV_CLUSTER_DETAIL,
    CSV_CLUSTER_WATCH,
    WATCH_RULE_IDS,
)
from src.transformers import (
    read_wazuh_logs,
    flatten_and_clean_logs,
    apply_feature_engineering,
    deduplicate,
    filter_period,
)
from src.clustering import (
    compute_elbow,
    compute_silhouette,
    plot_elbow,
    plot_silhouette,
)

from src.labeling import export_rule_labels

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


# ── Input Interaktif ───────────────────────────────────────────────────

def prompt_data_source() -> str:
    """Minta path arsip/folder log lalu salin ke data/raw/. ENTER = pakai isi data/raw/ yang ada."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        raw = input(
            "\nMasukkan path arsip (.tar.gz) atau folder log\n"
            "(ENTER = pakai isi data/raw/ yang sudah ada): "
        ).strip().strip('"').strip("'")

        if not raw:
            if any(DATA_RAW_DIR.iterdir()):
                return "data/raw (sudah ada)"
            print("data/raw/ kosong, isi path terlebih dahulu.")
            continue

        src = Path(raw)
        if not src.exists():
            print("Path tidak ditemukan, coba lagi.")
            continue

        src_resolved = src.resolve()
        raw_resolved = DATA_RAW_DIR.resolve()
        if src_resolved == raw_resolved or raw_resolved in src_resolved.parents:
            return str(src)  # sudah berada di data/raw/

        if src.is_file():
            shutil.copy2(src, DATA_RAW_DIR / src.name)
        else:
            shutil.copytree(src, DATA_RAW_DIR / src.name, dirs_exist_ok=True)
        log.info(f"Data disalin ke {DATA_RAW_DIR}: {src.name}")
        return str(src)


def prompt_period(df: pd.DataFrame):
    """Pilih periode data latih. Format: 2026 | 2026-03 | 2026-03-15."""
    monthly = df["timestamp"].dt.to_period("M").astype(str).value_counts().sort_index()
    print("\nJumlah log per bulan (semua agen):")
    for m, c in monthly.items():
        print(f"  {m}: {c:,}")
    print("\nFormat: 2026 | 2026-03 | 2026-03-15. Kosongkan keduanya = pakai semua data.")
    while True:
        start = input("Periode LATIH - awal  : ").strip() or None
        end = input("Periode LATIH - akhir (ENTER = sama dengan awal): ").strip() or None
        try:
            out = filter_period(df, start, end)
        except ValueError:
            print("Format tidak valid, coba lagi.")
            continue
        if out.empty:
            print("Tidak ada data pada periode itu.")
            continue
        return start, end


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

    selected_tokens = [token.strip() for token in user_input.split(",") if token.strip()]
    blacklist = set()

    for token in selected_tokens:
        if token in agent_map_by_index:
            blacklist.add(agent_map_by_index[token])
        elif token.lower() in agent_map_by_lower:
            blacklist.add(agent_map_by_lower[token.lower()])
        else:
            log.warning(f"Input '{token}' tidak cocok dengan agen mana pun, diabaikan.")

    if blacklist:
        log.info(f"Agen final yang masuk daftar blacklist: {sorted(blacklist)}")
        initial_count = len(df)
        df_filtered = df[~df["agent_name"].isin(blacklist)].copy()
        removed_count = initial_count - len(df_filtered)
        log.info(f"Data tersisa: {len(df_filtered):,} baris (Dibuang: {removed_count:,} baris)")
        return df_filtered

    log.info("Tidak ada agen valid yang dipilih untuk dibuang.")
    return df


def prompt_k(suggested: int) -> int:
    """Minta K dari user; ENTER = pakai K yang disarankan."""
    valid_ks = list(K_RANGE)
    while True:
        raw = input(
            f"\nPilih K ({valid_ks[0]}-{valid_ks[-1]}) "
            f"[saran: {suggested}, ENTER = pakai saran]: "
        ).strip()
        if not raw:
            return suggested
        if raw.isdigit() and int(raw) in valid_ks:
            return int(raw)
        print("Input tidak valid.")


# ── Profil & Template Pemetaan ─────────────────────────────────────────

def generate_cluster_profiles(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """Mengekstrak profil ringkas per cluster untuk analisis pakar."""
    profiles = []
    for cid in sorted(df_clustered["cluster"].unique()):
        subset = df_clustered[df_clustered["cluster"] == cid]
        profiles.append({
            "cluster": cid,
            "total_pola_dedup": len(subset),
            "total_alert_raw": subset["alert_count"].sum() if "alert_count" in subset.columns else len(subset),
            "rule_id_dominan": subset["rule_id"].mode()[0],
            "rule_level_dominan": subset["rule_level"].mode()[0],
            "decoder_dominan": subset["decoder_name"].mode()[0],
            "location_dominan": subset["location"].mode()[0],
            "hour_group_dominan": subset["hour_group"].mode()[0],
        })
    return pd.DataFrame(profiles)

def export_cluster_detail(df_clustered: pd.DataFrame, top_n: int = 8):
    """Isi tiap klaster (top rule + persentase) dan posisi rule pantauan, untuk bantu pelabelan."""
    rows = []
    for cid in sorted(df_clustered["cluster"].unique()):
        sub = df_clustered[df_clustered["cluster"] == cid]
        total = sub["alert_count"].sum()
        top = (sub.groupby(["rule_id", "rule_description", "rule_level"])["alert_count"]
                  .sum().sort_values(ascending=False).head(top_n).reset_index())
        top.insert(0, "cluster", cid)
        top["persen_cluster"] = (top["alert_count"] / total * 100).round(1)
        rows.append(top)
    pd.concat(rows, ignore_index=True).to_csv(CSV_CLUSTER_DETAIL, index=False)
    log.info(f"Detail klaster disimpan di: {CSV_CLUSTER_DETAIL}")

    watch = df_clustered[df_clustered["rule_id"].isin(WATCH_RULE_IDS)]
    if watch.empty:
        log.info("Tidak ada rule pantauan (WATCH_RULE_IDS) pada data periode ini.")
        return
    pivot = watch.pivot_table(
        index=["rule_id", "rule_description"], columns="cluster",
        values="alert_count", aggfunc="sum", fill_value=0,
    )
    pivot.to_csv(CSV_CLUSTER_WATCH)
    log.info(f"Posisi rule pantauan disimpan di: {CSV_CLUSTER_WATCH}")


# ── Pipeline Utama ─────────────────────────────────────────────────────

def run_pipeline():
    start = datetime.now()
    log.info("=" * 60)
    log.info("Script A: Pipeline Ekstraksi, Clustering & Profiling Dimulai")
    log.info(f"Waktu Mulai: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Sumber Data & Ekstraksi Log ─────────────────────────────
    log.info("Step 1: Menentukan sumber data & membaca log Wazuh...")
    source = prompt_data_source()
    raw_records = read_wazuh_logs(DATA_RAW_DIR)
    log.info(f"Total log mentah terbaca: {len(raw_records):,} baris")

    if not raw_records:
        log.error("Tidak ada data log yang berhasil dimuat. Proses dihentikan.")
        return

    # ── Step 2: Flattening, Periode & Blacklist ─────────────────────────
    log.info("Step 2: Meratakan struktur nested JSON, memilih periode & agen...")
    df = flatten_and_clean_logs(raw_records)
    if df.empty:
        log.error("Data kosong setelah cleaning. Proses dihentikan.")
        return

    p_start, p_end = prompt_period(df)
    df = filter_period(df, p_start, p_end)
    log.info(f"Periode latih: {p_start or 'semua'} s/d {p_end or p_start or 'semua'} ({len(df):,} baris)")

    agents_detected = sorted(df["agent_name"].unique().tolist())
    df = filter_blacklist_agents(df)

    if df.empty:
        log.error("Data kosong setelah blacklist. Proses dihentikan.")
        return

    agents_blacklisted = sorted(set(agents_detected) - set(df["agent_name"].unique()))
    rows_after_blacklist = len(df)

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

    # ── Step 4: Pemilihan K & K-Modes Clustering ────────────────────────
    log.info("Step 4: Pencarian K optimal (Elbow + Silhouette)...")
    n_distinct = len(deduped[CLUSTER_FEATURES].drop_duplicates())
    if n_distinct < max(K_RANGE):
        log.error(f"Kombinasi fitur unik ({n_distinct}) kurang dari K maksimum ({max(K_RANGE)}).")
        return

    sample = deduped.sample(
        n=min(K_SEARCH_SAMPLE, len(deduped)), random_state=KMODES_RANDOM_STATE
    )
    X_sample = sample[CLUSTER_FEATURES].astype(str).values

    costs = compute_elbow(X_sample)
    scores = compute_silhouette(X_sample)
    plot_elbow(costs)
    plot_silhouette(scores)

    df_k = pd.DataFrame({"k": list(K_RANGE), "cost": costs, "silhouette": scores})
    df_k.to_csv(CSV_K_SELECTION, index=False)
    print("\n" + df_k.to_string(index=False))

    valid = df_k.dropna(subset=["silhouette"])
    k_suggested = int(valid.loc[valid["silhouette"].idxmax(), "k"]) if not valid.empty else min(K_RANGE)
    k_chosen = prompt_k(k_suggested)
    log.info(f"K disarankan (silhouette tertinggi): {k_suggested} | K dipilih: {k_chosen}")

    log.info(f"Menjalankan K-Modes final (K={k_chosen}) pada fitur: {CLUSTER_FEATURES}...")
    X = deduped[CLUSTER_FEATURES].astype(str).values

    km = KModes(
        n_clusters=k_chosen,
        init=KMODES_INIT,
        n_init=KMODES_N_INIT,
        random_state=KMODES_RANDOM_STATE,
    )
    deduped["cluster"] = km.fit_predict(X)

    joblib.dump(km, MODEL_KMODES_FILE)
    log.info(f"Model K-Modes disimpan di: {MODEL_KMODES_FILE}")

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

    rule_table = export_rule_labels(deduped)
    log.info(f"Tabel label rule disimpan di: {CSV_RULE_LABELS} ({len(rule_table)} rule)")
    export_cluster_detail(deduped)

    # ── Step 6: Catatan Konfigurasi Run (dokumentasi laporan) ───────────
    run_config = {
        "tanggal_run": start.strftime("%Y-%m-%d %H:%M:%S"),
        "sumber_data": source,
        "periode_latih": {"awal": p_start, "akhir": p_end or p_start},
        "total_log_mentah": len(raw_records),
        "agen_terdeteksi": agents_detected,
        "agen_diblacklist": agents_blacklisted,
        "total_log_setelah_blacklist": rows_after_blacklist,
        "time_window_menit": TIME_WINDOW_MINUTES,
        "pola_unik_setelah_dedup": len(deduped),
        "fitur_clustering": CLUSTER_FEATURES,
        "k_range": [min(K_RANGE), max(K_RANGE)],
        "sampel_pencarian_k": len(sample),
        "k_disarankan": k_suggested,
        "k_dipilih": k_chosen,
        "jumlah_rule": len(rule_table),
        "distribusi_label_saran": rule_table.groupby("label_saran")["alert"].sum().to_dict(),
        "kmodes": {
            "init": KMODES_INIT,
            "n_init": KMODES_N_INIT,
            "random_state": KMODES_RANDOM_STATE,
        },
        "hasil_pencarian_k": df_k.astype(object).where(df_k.notna(), None).to_dict(orient="records"),
    }
    with open(CONFIG_RUN_JSON, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Konfigurasi run disimpan di: {CONFIG_RUN_JSON}")

    duration = int((datetime.now() - start).total_seconds())
    log.info("=" * 60)
    log.info(f"Script A selesai dalam {duration} detik.")
    print("\n" + "#" * 60)
    print("PROSES SELESAI: JEDA UNTUK HUMAN-IN-THE-LOOP")
    print(f"1. Buka '{CSV_PROFILES}' dan '{CSV_CLUSTER_DETAIL}' untuk melihat isi tiap klaster.")
    print(f"2. Buka '{CSV_RULE_LABELS}': kolom label_final sudah terisi dari label_saran.")
    print("   Ubah label_final bila perlu (koreksi tidak akan tertimpa saat Script A diulang).")
    print("3. Lanjutkan ke Script B (pelatihan), lalu Script C (evaluasi).")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline()