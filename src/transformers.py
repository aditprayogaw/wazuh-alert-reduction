# src/transformers.py
import gzip
import json
import tarfile
from pathlib import Path
import pandas as pd

from config import (
    ARCHIVE_FILE,
    TARGET_MONTH,
    JSON_GZ_PATTERN,
    EXTRACT_FIELDS,
    DEDUP_GROUP_COLS,
    TIME_WINDOW_MINUTES,
    HOUR_GROUPS,
    ALLOWED_AGENTS
)

# ── Helpers ────────────────────────────────────────────────────────────

def _get_nested(record: dict, path: list):
    """Ambil nilai dari nested dict berdasarkan list key secara aman."""
    val = record
    for key in path:
        if not isinstance(val, dict):
            return None
        val = val.get(key)
    return val


def _parse_record(record: dict, fields: dict) -> dict:
    """Flatten satu JSON record sesuai dengan fields yang dikonfigurasi."""
    return {
        col: _get_nested(record, path)
        for col, path in fields.items()
    }


def _iter_json_gz(tar: tarfile.TarFile, member: tarfile.TarInfo):
    """Generator: yield satu parsed dict per baris dari file .json.gz."""
    f = tar.extractfile(member)
    if f is None:
        return
    with gzip.open(f, "rt", encoding="utf-8", errors="replace") as gz:
        for line in gz:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ── Data Loading ───────────────────────────────────────────────────────

def load_all_months() -> pd.DataFrame:
    """
    Baca semua ossec-alerts-XX.json.gz dari arsip untuk kebutuhan analisis tren makro.
    Fungsi ini membaca field bertingkat asli JSON menggunakan titik (.) sebelum dirapikan.
    """
    records = []
    with tarfile.open(ARCHIVE_FILE, "r:gz") as tar:
        for member in tar.getmembers():
            p = Path(member.name)
            if not p.name.endswith(".json.gz") or not p.match(JSON_GZ_PATTERN):
                continue
            
            # Mengambil nama folder bulan (misal: May, Jun, Jul)
            month = p.parts[-2] if len(p.parts) >= 2 else "unknown"
            
            for record in _iter_json_gz(tar, member):
                # Ambil data mentah menggunakan key bertingkat asli dari objek JSON
                agent_obj = record.get("agent", {})
                rule_obj = record.get("rule", {})
                
                records.append({
                    "timestamp": record.get("timestamp"),
                    "agent_name": agent_obj.get("name") if isinstance(agent_obj, dict) else None,
                    "rule_level": rule_obj.get("level") if isinstance(rule_obj, dict) else None,
                    "month": month
                })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["rule_level"] = pd.to_numeric(df["rule_level"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


def load_june_logs() -> pd.DataFrame:
    """
    Baca semua file log jalur folder target (Juni).
    Mengekstrak data mentah tanpa mengubah nama field bertingkat aslinya 
    untuk kebutuhan audit fase Data Understanding.
    """
    records = []
    with tarfile.open(ARCHIVE_FILE, "r:gz") as tar:
        for member in tar.getmembers():
            p = Path(member.name)
            if TARGET_MONTH not in p.parts:
                continue
            if not p.name.endswith(".json.gz") or not p.match(JSON_GZ_PATTERN):
                continue
            for record in _iter_json_gz(tar, member):
                records.append(_parse_record(record, EXTRACT_FIELDS))
    return pd.DataFrame(records)


# ── Cleaning & Flattening (Data Preparation) ───────────────────────────
def clean_logs(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Eksekusi transformasi data mentah (df_raw) menjadi berformat flat tabel dua dimensi (df).
    Melakukan penanganan nilai kosong (drop null), penyelarasan tipe data resmi,
    dan memfilter agent yang tidak relevan dengan penelitian.
    """
    # 1. Drop null pada field kritis
    critical_raw_cols = ["timestamp", "agent_name", "rule_level", "rule_id", "rule_description", "decoder_name"]
    df_clean = df_raw.dropna(subset=critical_raw_cols).copy()

    # 2. Konversi tipe data
    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce")
    df_clean["agent_name"] = df_clean["agent_name"].astype(str)
    df_clean["rule_level"] = pd.to_numeric(df_clean["rule_level"], errors="coerce").astype(int)
    df_clean["rule_description"] = df_clean["rule_description"].astype(str)
    df_clean["rule_id"] = df_clean["rule_id"].astype(str)
    df_clean["rule_groups"] = df_clean["rule_groups"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else str(x) if pd.notna(x) else "unknown"
    )
    df_clean["decoder_name"] = df_clean["decoder_name"].astype(str)
    df_clean["location"] = df_clean["location"].astype(str)

    # 3. Filter 4 agent yang relevan
    df_clean = df_clean[df_clean["agent_name"].isin(ALLOWED_AGENTS)]

    # 4. Drop timestamp null
    df_clean = df_clean.dropna(subset=["timestamp"])

    return df_clean.reset_index(drop=True)


# ── Feature Engineering (Data Preparation) ────────────────────────────────

def get_hour_group(hour: int) -> str:
    """Transformasi nilai jam numerik (0-23) ke kelompok teks temporal siber."""
    for group_name, hour_range in HOUR_GROUPS.items():
        if hour in hour_range:
            return group_name
    return "malam"

    
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Rekayasa Fitur Tahap 1: Ekstraksi waktu, tanggal, dan grup jam."""
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["hour_group"] = df["hour"].apply(get_hour_group)
    return df


# ── Time-Window Deduplication (Data Preparation) ────────────────────────
def deduplicate_time_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aglomerasi Kuantitas Log Berbasis Jendela Waktu (Time-Window Aggregation).
    Menggabungkan spam log sejenis dalam interval menit yang ditentukan di config.py
    """
    df = df.copy()

    # 1. Membuat penanda interval jendela waktu (floor rounding)
    df["time_window"] = df["timestamp"].dt.floor(f"{TIME_WINDOW_MINUTES}min")

    # 2. Menggabungkan kolom kunci penentu dengan kolom jendela waktu
    group_keys = DEDUP_GROUP_COLS + ["time_window"]

    # 3. Proses kompresi volume log melalui teknik agregasi kuantitas
    deduped = df.groupby(group_keys).size().reset_index(name="alert_count")

    # 4. Ekstraksi ulang fitur jam dari kolom jendela waktu yang baru terbentuk
    deduped["hour"] = deduped["time_window"].dt.hour
    deduped["hour_group"] = deduped["hour"].apply(get_hour_group)

    # 5. Kembalikan dataframe dengan membuang kolom penanda waktu sementara
    return deduped.drop(columns=["time_window", "hour"])

