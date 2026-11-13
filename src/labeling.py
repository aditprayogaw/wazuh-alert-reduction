# src/labeling.py
"""
Manajemen Katalog Kebijakan Rule (Tabel Kebijakan Triase SOC).

Menyediakan utilitas untuk:
1. Membangun katalog rule unik dari data yang telah terdeduplikasi.
2. Mengekspor template tabel kebijakan (data/processed/rule_labels.csv) untuk pelabelan manual:
   - Dismiss : Tidak perlu tindakan analis.
   - Review  : Ambigu atau indikasi upaya serangan tanpa tanda berhasil (termasuk 4xx).
   - Escalate: Indikasi insiden nyata, pembuatan file direktori web, atau dampak kritis.
3. Membaca tabel kebijakan untuk pemetaan deterministik pada fase klasifikasi/triase.
"""
from pathlib import Path
import pandas as pd
from config import CSV_RULE_LABELS


def build_rule_catalog(df_deduped: pd.DataFrame) -> pd.DataFrame:
    """
    Ekstrak daftar rule unik beserta atribut penjelas dari data terdeduplikasi.
    """
    # Agregasi informasi per rule_id
    grouped = df_deduped.groupby("rule_id", as_index=False).agg(
        rule_level=("rule_level", "first"),
        decoder_name=("decoder_name", lambda s: s.mode()[0] if not s.empty else "-"),
        location=("location", lambda s: s.mode()[0] if not s.empty else "-"),
        rule_groups=("rule_groups", "first"),
        rule_description=("rule_description", "first"),
        total_kemunculan=("alert_count", "sum") if "alert_count" in df_deduped.columns else ("rule_id", "size"),
    )

    grouped["rule_id"] = grouped["rule_id"].astype(str)
    # Urutkan berdasarkan kemunculan terbanyak
    grouped = grouped.sort_values("total_kemunculan", ascending=False).reset_index(drop=True)
    return grouped


def export_rule_labels(df_deduped: pd.DataFrame) -> pd.DataFrame:
    """
    Tulis atau perbarui data/processed/rule_labels.csv.
    Koreksi/isi manual label_final yang sudah ada pada file lama tidak akan tertimpa.
    """
    catalog = build_rule_catalog(df_deduped)
    catalog["label_final"] = ""  # Diisi manual: Dismiss, Review, Escalate
    catalog["catatan_alasan"] = ""

    # Jika file sudah ada sebelumnya, pertahankan label dan catatan yang sudah diisi
    if CSV_RULE_LABELS.exists():
        try:
            old = pd.read_csv(CSV_RULE_LABELS, dtype={"rule_id": str})
            kept_labels = {}
            kept_notes = {}
            for _, row in old.iterrows():
                rid = str(row.get("rule_id", "")).strip()
                lbl = str(row.get("label_final", "")).strip()
                note = str(row.get("catatan_alasan", "")).strip()
                if lbl and lbl != "nan":
                    kept_labels[rid] = lbl
                if note and note != "nan":
                    kept_notes[rid] = note

            catalog["label_final"] = catalog["rule_id"].map(lambda r: kept_labels.get(r, ""))
            catalog["catatan_alasan"] = catalog["rule_id"].map(lambda r: kept_notes.get(r, ""))
        except Exception:
            pass

    CSV_RULE_LABELS.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(CSV_RULE_LABELS, index=False)
    return catalog


def load_rule_labels() -> dict:
    """
    Membaca tabel kebijakan rule_labels.csv -> mengembalikan mapping {rule_id: label_final}.
    Memvalidasi bahwa seluruh rule telah memiliki label aksi triase.
    """
    if not CSV_RULE_LABELS.exists():
        raise FileNotFoundError(f"{CSV_RULE_LABELS} belum ada. Jalankan scripts/prepare_data.py terlebih dahulu.")

    table = pd.read_csv(CSV_RULE_LABELS, dtype={"rule_id": str})
    table["label_final"] = table["label_final"].fillna("").astype(str).str.strip()

    unlabeled = table.loc[table["label_final"] == "", "rule_id"].tolist()
    if unlabeled:
        raise ValueError(
            f"Masih ada {len(unlabeled)} rule yang belum memiliki label_final: {unlabeled}. "
            f"Lengkapi file {CSV_RULE_LABELS} terlebih dahulu."
        )

    # Validasi hanya nilai label yang sah
    valid_labels = {"Dismiss", "Review", "Escalate"}
    invalid = table[~table["label_final"].isin(valid_labels)]
    if not invalid.empty:
        contoh = invalid[["rule_id", "label_final"]].to_dict(orient="records")
        raise ValueError(f"Ditemukan label tidak valid di {CSV_RULE_LABELS}: {contoh}. Hanya gunakan: Dismiss, Review, Escalate.")

    return dict(zip(table["rule_id"], table["label_final"]))
