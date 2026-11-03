# src/transformers.py
import gzip
import json
import tarfile
from pathlib import Path
import pandas as pd

from config import (
    ARCHIVE_EXTENSION,
    JSON_GZ_PATTERN,
    EXTRACT_FIELDS,
    DEDUP_GROUP_COLS,
    TIME_WINDOW_MINUTES,
    HOUR_GROUPS,
)

LOCAL_TZ = "Asia/Makassar"  # WITA (Bali)


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
    """Generator: yield satu parsed dict per baris dari file .json.gz di dalam tar."""
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


def _iter_json_gz_file(path: Path):
    """Generator: baca satu file .json.gz biasa (di folder)."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as gz:
        for line in gz:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ── Data Loading ───────────────────────────────────────────────────────

def read_wazuh_logs(raw_dir: Path) -> list:
    """Baca semua ossec-alerts-*.json.gz dari arsip .tar.gz maupun folder bertingkat."""
    records = []

    for archive in sorted(raw_dir.rglob(f"*{ARCHIVE_EXTENSION}")):
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                p = Path(member.name)
                if not member.isfile() or not p.match(JSON_GZ_PATTERN):
                    continue
                for record in _iter_json_gz(tar, member):
                    records.append(_parse_record(record, EXTRACT_FIELDS))

    for gz_file in sorted(raw_dir.rglob(JSON_GZ_PATTERN)):
        for record in _iter_json_gz_file(gz_file):
            records.append(_parse_record(record, EXTRACT_FIELDS))

    return records


# ── Cleaning & Flattening (Data Preparation) ───────────────────────────

def flatten_and_clean_logs(records: list) -> pd.DataFrame:
    """Ubah list record menjadi tabel 2D bersih (drop null kolom esensial)."""
    df_raw = pd.DataFrame(records)
    if df_raw.empty:
        return df_raw

    critical_cols = ["timestamp", "agent_name", "rule_level", "rule_id",
                     "rule_groups", "decoder_name", "location"]
    df_clean = df_raw.dropna(subset=critical_cols).copy()

    # Timestamp: parse ke UTC (aman untuk offset campuran), konversi ke WITA, buang tz
    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce", utc=True)
    df_clean = df_clean.dropna(subset=["timestamp"])
    df_clean["timestamp"] = df_clean["timestamp"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

    df_clean["agent_name"] = df_clean["agent_name"].astype(str)
    df_clean["rule_level"] = pd.to_numeric(df_clean["rule_level"], errors="coerce")
    df_clean = df_clean.dropna(subset=["rule_level"])
    df_clean["rule_level"] = df_clean["rule_level"].astype(int)
    df_clean["rule_description"] = df_clean["rule_description"].fillna("-").astype(str)
    df_clean["rule_id"] = df_clean["rule_id"].astype(str)
    df_clean["rule_groups"] = df_clean["rule_groups"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else str(x)
    )
    df_clean["decoder_name"] = df_clean["decoder_name"].astype(str)
    df_clean["location"] = df_clean["location"].astype(str)

    return df_clean.reset_index(drop=True)


# ── Feature Engineering (Data Preparation) ─────────────────────────────

def get_hour_group(hour: int) -> str:
    """Transformasi nilai jam numerik (0-23) ke kelompok teks temporal."""
    for group_name, hour_range in HOUR_GROUPS.items():
        if hour in hour_range:
            return group_name
    return "malam"


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Rekayasa fitur: ekstraksi tanggal, jam, dan grup jam."""
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["hour_group"] = df["hour"].apply(get_hour_group)
    return df


# ── Time-Window Deduplication (Data Preparation) ───────────────────────

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Agregasi log sejenis per jendela waktu; alert_count = jumlah kemunculan."""
    df = df.copy()
    df["time_window"] = df["timestamp"].dt.floor(f"{TIME_WINDOW_MINUTES}min")

    group_keys = DEDUP_GROUP_COLS + ["time_window"]

    deduped = (
        df.groupby(group_keys, as_index=False)
          .agg(
              alert_count=("rule_id", "size"),
              rule_description=("rule_description", "first"),
              rule_groups=("rule_groups", "first"),
          )
    )

    deduped["hour_group"] = deduped["time_window"].dt.hour.apply(get_hour_group)
    return deduped.drop(columns=["time_window"])
    
# ── Filter Periode ─────────────────────────────────────────────────────

def period_bounds(text: str):
    """'2026' | '2026-03' | '2026-03-15' -> (awal, akhir_eksklusif). Error: ValueError."""
    p = pd.Period(text.strip())
    return p.start_time, (p + 1).start_time


def filter_period(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Filter baris berdasarkan periode (waktu WITA). Kosong semua = semua data.
    end kosong = sama dengan periode start."""
    if not start and not end:
        return df
    lo = period_bounds(start)[0] if start else df["timestamp"].min()
    hi = period_bounds(end or start)[1]
    return df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)].copy()