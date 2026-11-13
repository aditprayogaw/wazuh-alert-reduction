# scripts/prepare_data.py
"""
Script 1: Persiapan Data & Ekspor Katalog Kebijakan Rule (Tahap 1 CRISP-DM / Triase SOC)

Fungsi Utama:
1. Memuat log Wazuh secara dinamis dari arsip data/raw/.
2. Pemilihan Periode Latih secara interaktif (Tahun, Bulan, atau Rentang Tanggal).
3. Pemilihan Agen Blacklist secara interaktif (nomor urut atau nama agen).
4. Pembersihan (Cleaning) dan Deduplikasi Temporal (10 menit).
5. Ekspor data bersih terdeduplikasi (data/processed/deduped.parquet).
6. Ekspor template katalog kebijakan rule (data/processed/rule_labels.csv) 
   untuk diisi manual oleh analis SOC (Dismiss / Review / Escalate).

Prinsip:
- Bersih dari label leakage (tidak ada ketergantungan pada klaster K-Modes).
- Fleksibel dan interaktif.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (
    DATA_RAW_DIR,
    DATA_PROC_DIR,
    REPORTS_DIR,
    PARQUET_CLEANED,
    PARQUET_DEDUPED,
    CSV_DEDUPED,
    CSV_RULE_LABELS,
    CONFIG_RUN_JSON,
    TIME_WINDOW_MINUTES,
)
from src.transformers import (
    read_wazuh_logs,
    flatten_and_clean_logs,
    apply_feature_engineering,
    deduplicate,
    filter_period,
)
from src.labeling import export_rule_labels

# ── Logging Setup ──────────────────────────────────────────────────────
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


# ── Modul Interaktif ───────────────────────────────────────────────────

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
    print("\n" + "=" * 55)
    print("DISTRIBUSI DATA LOG PER BULAN (Semua Agen):")
    print("=" * 55)
    for m, c in monthly.items():
        print(f"  {m:<10} : {c:>10,} alert")
    print("-" * 55)
    print("Format input: '2026-03' (bulan) atau '2026-03-01' s.d. '2026-03-31'")
    print("(Tekan ENTER pada keduanya jika ingin menggunakan SEMUA data)")
    print("-" * 55)

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


def filter_blacklist_agents(df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    """
    Menampilkan semua agen yang terdeteksi dan memfilter berdasarkan input blacklist.
    Mendukung pemilihan via nomor urut (1, 2) maupun nama agen (case-insensitive).
    """
    detected_agents = sorted(df["agent_name"].dropna().unique().tolist())

    if not detected_agents:
        log.warning("Tidak ada agent_name yang ditemukan di dalam data.")
        return df, [], []

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
        "(Contoh: '1, 3' atau 'DVWA, dfir-iris, proxy-manager, soc-1' | Tekan ENTER jika tidak ada): "
    ).strip()

    if not user_input:
        log.info("Tidak ada agen yang di-blacklist.")
        return df, detected_agents, []

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
        blacklist_list = sorted(list(blacklist))
        log.info(f"Agen yang di-blacklist: {blacklist_list}")
        initial_count = len(df)
        df_filtered = df[~df["agent_name"].isin(blacklist)].copy()
        removed_count = initial_count - len(df_filtered)
        log.info(f"Data tersisa: {len(df_filtered):,} baris (Dibuang: {removed_count:,} baris)")
        return df_filtered, detected_agents, blacklist_list

    log.info("Tidak ada agen valid yang dipilih untuk dibuang.")
    return df, detected_agents, []


# ── Pipeline Utama ─────────────────────────────────────────────────────

def run_pipeline():
    start = datetime.now()
    log.info("=" * 65)
    log.info("SKRIP 1: Persiapan Data & Ekspor Katalog Kebijakan Rule Dimulai")
    log.info(f"Waktu Mulai: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 65)

    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Sumber Data Mentah
    source = prompt_data_source()

    # 2. Pembacaan dan Pembersihan Awal Log
    log.info("Membaca log Wazuh mentah secara dinamis...")
    raw_records = read_wazuh_logs(DATA_RAW_DIR)
    log.info(f"Total log mentah terbaca: {len(raw_records):,} baris")

    if not raw_records:
        log.error("Tidak ada data log yang berhasil dimuat. Proses dihentikan.")
        return

    log.info("Melakukan data flattening & cleaning awal...")
    df = flatten_and_clean_logs(raw_records)

    # 3. Pemilihan Periode Data
    p_start, p_end = prompt_period(df)
    df = filter_period(df, p_start, p_end)
    periode_label = f"{p_start} s.d. {p_end or p_start}" if p_start else "Semua Data"
    log.info(f"Periode terpilih ({periode_label}): {len(df):,} baris")

    # 4. Filter Blacklist Agen
    df, agents_detected, agents_blacklisted = filter_blacklist_agents(df)
    if df.empty:
        log.error("Data kosong setelah filter blacklist. Proses dihentikan.")
        return

    rows_after_blacklist = len(df)
    df.to_parquet(PARQUET_CLEANED, index=False)
    log.info(f"Data cleaned disimpan di: {PARQUET_CLEANED}")

    # 5. Feature Engineering & Deduplikasi Temporal (10 Menit)
    log.info(f"Menjalankan deduplikasi temporal ({TIME_WINDOW_MINUTES} menit)...")
    df = apply_feature_engineering(df)
    deduped = deduplicate(df)

    reduction_rate = (1 - (len(deduped) / len(df))) * 100
    log.info(f"Sebelum deduplikasi : {len(df):,} baris alert")
    log.info(f"Setelah deduplikasi : {len(deduped):,} pola unik")
    log.info(f"Tingkat Reduksi Data : {reduction_rate:.2f}%")

    deduped.to_parquet(PARQUET_DEDUPED, index=False)
    deduped.to_csv(CSV_DEDUPED, index=False)
    log.info(f"Data terdeduplikasi disimpan di: {PARQUET_DEDUPED}")

    # 6. Ekspor Katalog Kebijakan Rule (Tabel Pelabelan Manual)
    log.info("Mengekstrak katalog rule unik untuk Tabel Kebijakan Triase...")
    rule_table = export_rule_labels(deduped)
    log.info(f"Tabel kebijakan rule disimpan di: {CSV_RULE_LABELS} ({len(rule_table)} rule unik)")

    # 7. Simpan Catatan Konfigurasi Run
    run_config = {
        "tanggal_run": start.strftime("%Y-%m-%d %H:%M:%S"),
        "sumber_data": source,
        "periode_latih": {"awal": p_start, "akhir": p_end or p_start},
        "total_log_mentah": len(raw_records),
        "agen_terdeteksi": agents_detected,
        "agen_diblacklist": agents_blacklisted,
        "total_log_setelah_filter": rows_after_blacklist,
        "time_window_menit": TIME_WINDOW_MINUTES,
        "pola_unik_setelah_dedup": len(deduped),
        "tingkat_reduksi_persen": round(reduction_rate, 2),
        "jumlah_rule_unik": len(rule_table),
    }
    with open(CONFIG_RUN_JSON, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Catatan run disimpan di: {CONFIG_RUN_JSON}")

    duration = int((datetime.now() - start).total_seconds())
    log.info("=" * 65)
    log.info(f"Skrip 1 selesai dalam {duration} detik.")
    print("\n" + "#" * 65)
    print("PROSES SELESAI: SIAP UNTUK TAHAP 1 (HUMAN-IN-THE-LOOP LABELING)")
    print(f"1. Buka file: '{CSV_RULE_LABELS}'")
    print(f"   (Terdapat {len(rule_table)} baris rule unik)")
    print("2. Isi kolom 'label_final' dengan salah satu dari 3 aksi triase:")
    print("   - Dismiss  : Tidak butuh tindakan.")
    print("   - Review   : Ambigu / upaya tanpa bukti sukses (termasuk 4xx).")
    print("   - Escalate : Indikasi insiden nyata / dampak kritis.")
    print("3. (Opsional) Isi kolom 'catatan_alasan' jika ada pertimbangan khusus.")
    print("4. Setelah selesai melabeli, simpan file tersebut.")
    print("#" * 65 + "\n")


if __name__ == "__main__":
    run_pipeline()
