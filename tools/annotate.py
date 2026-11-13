# scripts/annotate.py
"""
Alat bantu pelabelan manual: lembar anotasi, audit pakar, dan penghitung kesepakatan.
Ini BUKAN tahap pipeline otomatis, melainkan bantuan untuk langkah manual di antara
scripts.prepare_data dan scripts.train_model.

Peran:
  - Peneliti melabeli SENDIRI seluruh sampel (Dismiss / Review / Escalate).
  - Pakar mengaudit sebagian hasilnya tanpa melihat label peneliti.
Padanan Indonesia: Abaikan = Dismiss, Tinjau = Review, Eskalasi = Escalate (tetap diterima).

Pemakaian (dari folder utama proyek), semua file ada di data/processed/anotasi/:
  1) python -m scripts.annotate sheet --n 300 --top 80 --pilot 40
       -> anotasi.csv         lembar kerja peneliti (isi kolom `label` dan `alasan`)
          anotasi_kunci.csv   kunci untuk peneliti, JANGAN dibagikan
  2) Peneliti mengisi anotasi.csv.
  3) python -m scripts.annotate audit
       -> audit_untuk_pakar.csv  semua Escalate + sebagian Dismiss/Review, label peneliti disembunyikan
  4) Pakar mengisi audit_untuk_pakar.csv (simpan di tempat yang sama), lalu:
       python -m scripts.annotate kappa
       -> persen sepakat per label, kappa, dan selisih_untuk_adjudikasi.csv

PERINGATAN: setelah pelabelan dimulai, jangan menjalankan ulang scripts.prepare_data dengan
pengaturan berbeda. ID pola (pola_id) terikat pada urutan data hasil prepare_data.

Desain sampel (karena hasil utama adalah reduksi alert, pola 'berat' harus terwakili):
  - top      : pola dengan alert_count terbesar (mewakili mayoritas beban alert)
  - berisiko : pola dari rule berlevel tinggi atau rule pantauan (WATCH_RULE_IDS)
  - acak     : sisanya, acak terstratifikasi per kategori rule
Pelabel TIDAK melihat label_saran, rule_groups, maupun klaster, supaya tidak terpengaruh.
"""
import argparse

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from config import PARQUET_CLUSTERED, DATA_PROC_DIR, WATCH_RULE_IDS, RF_RANDOM_STATE
from src.labeling import load_rule_labels

ANNOT_DIR = DATA_PROC_DIR / "anotasi"
SHEET_PATH = ANNOT_DIR / "anotasi.csv"
KEY_PATH = ANNOT_DIR / "anotasi_kunci.csv"
AUDIT_PATH = ANNOT_DIR / "audit_untuk_pakar.csv"
DIFF_PATH = ANNOT_DIR / "selisih_untuk_adjudikasi.csv"

VALID = ["Dismiss", "Review", "Escalate"]
ALIAS = {"Abaikan": "Dismiss", "Tinjau": "Review", "Eskalasi": "Escalate"}
EVIDENCE_COLS = ["pola_id", "rule_id", "rule_description", "rule_level", "agent_name",
                 "location", "decoder_name", "hour_group", "alert_count"]
OPTIONAL_COLS = ["window_start", "timestamp", "first_seen", "last_seen",
                 # konteks web (hanya ada bila data memuat url; kosong/0 untuk rule non-web)
                 "url_example", "n_src_ip", "n_url", "top_ip_share",
                 "share_2xx", "share_3xx", "share_4xx", "share_5xx"]
RISKY_LEVEL = 10


# ── Pembantu ───────────────────────────────────────────────────────────
def _read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"rule_id": str}, encoding="utf-8-sig").fillna("")


def _normalize(series: pd.Series) -> pd.Series:
    """Rapikan huruf dan terima padanan Indonesia."""
    return series.astype(str).str.strip().str.title().replace(ALIAS)


def _has_labels(path) -> bool:
    """True bila file ada dan kolom `label`-nya sudah ada yang terisi (pelindung dari tertimpa)."""
    if not path.exists():
        return False
    d = _read_csv(path)
    return "label" in d.columns and bool((d["label"].astype(str).str.strip() != "").any())


def landis_koch(k: float) -> str:
    if pd.isna(k):
        return "tidak terdefinisi"
    if k < 0:
        return "di bawah kebetulan"
    for batas, nama in [(0.20, "sangat lemah"), (0.40, "cukup"), (0.60, "sedang"),
                        (0.80, "substansial"), (1.00, "hampir sempurna")]:
        if k <= batas:
            return nama
    return "hampir sempurna"


# ── sheet: buat lembar anotasi ─────────────────────────────────────────
def cmd_sheet(args):
    if _has_labels(SHEET_PATH) and not args.force:
        print(f"{SHEET_PATH.name} sudah berisi label. Dihentikan agar labelmu tidak tertimpa. "
              "Gunakan --force hanya bila memang ingin membuat ulang dari nol.")
        return

    df = pd.read_parquet(PARQUET_CLUSTERED).reset_index(drop=True)
    df["rule_id"] = df["rule_id"].astype(str)
    df["pola_id"] = [f"P{i:05d}" for i in range(len(df))]
    df["kategori"] = df["rule_id"].map(load_rule_labels()).fillna("?")

    # 1) pola terberat
    top = df.nlargest(args.top, "alert_count").assign(strata="top")

    # 2) pola berisiko: level tinggi atau rule pantauan, maksimal per_risky pola per rule
    lvl = pd.to_numeric(df["rule_level"], errors="coerce")
    risky_mask = (lvl >= RISKY_LEVEL) | df["rule_id"].isin([str(r) for r in WATCH_RULE_IDS])
    risky = (df[risky_mask].sort_values("alert_count", ascending=False)
             .groupby("rule_id").head(args.per_risky).assign(strata="berisiko"))

    chosen = pd.concat([top, risky]).drop_duplicates("pola_id")

    # 3) sisanya acak terstratifikasi per kategori
    n_rest = max(0, args.n - len(chosen))
    rest = df[~df["pola_id"].isin(chosen["pola_id"])]
    parts = []
    if n_rest > 0 and len(rest) > 0:
        for _, g in rest.groupby("kategori"):
            k = min(len(g), max(1, round(n_rest * len(g) / len(rest))))
            parts.append(g.sample(k, random_state=RF_RANDOM_STATE))
    rest_s = (pd.concat(parts).sample(frac=1, random_state=RF_RANDOM_STATE).head(n_rest)
              .assign(strata="acak")) if parts else rest.iloc[0:0].assign(strata="acak")

    sample = (pd.concat([chosen, rest_s])
              .sample(frac=1, random_state=RF_RANDOM_STATE).reset_index(drop=True))
    sample["pilot"] = (sample.index < args.pilot).astype(int)

    cols = [c for c in ["pilot"] + EVIDENCE_COLS + OPTIONAL_COLS if c in sample.columns]
    sheet = sample[cols].copy()
    sheet["label"] = ""
    sheet["alasan"] = ""

    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(SHEET_PATH, index=False, encoding="utf-8-sig")
    key_cols = [c for c in ["pola_id", "strata", "kategori", "alert_count", "rule_groups"]
                if c in sample.columns]
    sample[key_cols].to_csv(KEY_PATH, index=False, encoding="utf-8-sig")

    share = sample["alert_count"].sum() / df["alert_count"].sum() * 100
    print(f"Sampel: {len(sample)} pola (pilot: {int(sample['pilot'].sum())})")
    print(sample["strata"].value_counts().to_string())
    print(f"Sampel mewakili {share:.1f}% dari total alert mentah pada data ini.")
    print(f"Lembar kerja : {SHEET_PATH}")
    print(f"Kunci        : {KEY_PATH} (jangan dibagikan)")


# ── audit: lembar untuk pakar ──────────────────────────────────────────
def cmd_audit(args):
    if _has_labels(AUDIT_PATH) and not args.force:
        print(f"{AUDIT_PATH.name} sudah diisi pakar. Dihentikan agar isiannya tidak tertimpa. "
              "Gunakan --force hanya bila memang ingin membuat ulang.")
        return

    d = _read_csv(args.file)
    d["label"] = _normalize(d["label"])
    bad = d[~d["label"].isin(VALID)]
    if len(bad):
        print(f"PERINGATAN: {len(bad)} baris belum berlabel atau tidak valid dan dilewati. "
              "Selesaikan pelabelan dulu.")
        d = d[d["label"].isin(VALID)]

    frac = {"Escalate": 1.0, "Dismiss": args.dismiss_frac, "Review": args.review_frac}
    parts = []
    for lab, g in d.groupby("label"):
        k = len(g) if frac[lab] >= 1 else max(1, round(len(g) * frac[lab]))
        parts.append(g.sample(min(k, len(g)), random_state=RF_RANDOM_STATE))
    if not parts:
        print("Tidak ada baris berlabel.")
        return

    aud = pd.concat(parts).sample(frac=1, random_state=RF_RANDOM_STATE)
    cols = [c for c in EVIDENCE_COLS + OPTIONAL_COLS if c in aud.columns]
    out = aud[cols].copy()
    out["label"] = ""
    out["alasan"] = ""
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(AUDIT_PATH, index=False, encoding="utf-8-sig")
    print(f"Lembar audit: {len(out)} pola (label peneliti disembunyikan dari pakar)")
    print(aud["label"].value_counts().to_string())
    print(f"File: {AUDIT_PATH}")


# ── kappa: kesepakatan peneliti vs pakar ───────────────────────────────
def _report(m: pd.DataFrame, judul: str):
    k = cohen_kappa_score(m["label_peneliti"], m["label_pakar"], labels=VALID)
    kw = cohen_kappa_score(m["label_peneliti"], m["label_pakar"], labels=VALID, weights="linear")
    agree = (m["label_peneliti"] == m["label_pakar"]).mean() * 100
    print(f"\n== {judul} (n={len(m)}) ==")
    print(f"Persen sepakat       : {agree:.1f}%")
    print(f"Cohen's kappa        : {k:.3f} ({landis_koch(k)})")
    print(f"Kappa berbobot linier: {kw:.3f} (selisih Dismiss-Escalate dihitung lebih berat)")
    print(pd.crosstab(m["label_peneliti"], m["label_pakar"])
          .reindex(index=VALID, columns=VALID, fill_value=0))


def cmd_kappa(args):
    a = _read_csv(args.peneliti)
    b = _read_csv(args.pakar)
    a["label"] = _normalize(a["label"])
    b["label"] = _normalize(b["label"])
    keep = [c for c in ["pola_id", "pilot", "rule_id", "rule_description",
                        "alert_count", "label", "alasan"] if c in a.columns]
    m = a[keep].merge(b[["pola_id", "label", "alasan"]], on="pola_id",
                      suffixes=("_peneliti", "_pakar"))

    invalid = m[~m["label_peneliti"].isin(VALID) | ~m["label_pakar"].isin(VALID)]
    if len(invalid):
        print(f"PERINGATAN: {len(invalid)} baris dilewati (label kosong/tidak valid). "
              f"Label yang sah: {VALID}")
    m = m.drop(invalid.index)
    if m.empty:
        print("Tidak ada baris yang bisa dihitung.")
        return

    if "pilot" in m.columns and (m["pilot"].astype(str) == "1").any():
        _report(m[m["pilot"].astype(str) == "1"], "Pilot saja")
    _report(m, "Semua baris yang diaudit")

    per = (m.assign(sepakat=(m["label_peneliti"] == m["label_pakar"]))
             .groupby("label_peneliti")["sepakat"].agg(["size", "mean"]))
    per["mean"] = (per["mean"] * 100).round(1)
    print("\nKesepakatan per label peneliti (n pola | % sama dengan pakar):")
    print(per.rename(columns={"size": "n", "mean": "%_sepakat"}).to_string())
    print("Catatan: sampel audit dipilih berdasarkan label peneliti, sehingga kappa bias. "
          "Utamakan persen sepakat per label.")

    diff = m[m["label_peneliti"] != m["label_pakar"]].copy()
    diff["label_final"] = ""
    cols = [c for c in ["pola_id", "rule_id", "rule_description", "alert_count",
                        "label_peneliti", "label_pakar", "alasan_peneliti", "alasan_pakar",
                        "label_final"] if c in diff.columns]
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    diff[cols].to_csv(DIFF_PATH, index=False, encoding="utf-8-sig")
    print(f"\n{len(diff)} selisih disimpan di: {DIFF_PATH} (isi kolom label_final setelah dibahas)")


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(prog="python -m scripts.annotate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sheet", help="buat lembar anotasi")
    s.add_argument("--n", type=int, default=300, help="target jumlah pola (default 300)")
    s.add_argument("--top", type=int, default=80, help="jumlah pola terberat (default 80)")
    s.add_argument("--per-risky", dest="per_risky", type=int, default=10,
                   help="maks pola per rule berisiko (default 10)")
    s.add_argument("--pilot", type=int, default=40, help="jumlah pola pilot (default 40)")
    s.add_argument("--force", action="store_true", help="timpa lembar yang sudah berlabel")
    s.set_defaults(func=cmd_sheet)

    a = sub.add_parser("audit", help="buat lembar audit untuk pakar dari lembar yang sudah dilabeli")
    a.add_argument("file", nargs="?", default=str(SHEET_PATH),
                   help=f"lembar peneliti yang sudah diisi (default {SHEET_PATH})")
    a.add_argument("--dismiss-frac", dest="dismiss_frac", type=float, default=0.2,
                   help="porsi Dismiss yang diaudit (default 0.2)")
    a.add_argument("--review-frac", dest="review_frac", type=float, default=0.1,
                   help="porsi Review yang diaudit (default 0.1)")
    a.add_argument("--force", action="store_true", help="timpa lembar audit yang sudah diisi pakar")
    a.set_defaults(func=cmd_audit)

    k = sub.add_parser("kappa", help="hitung kesepakatan peneliti vs pakar")
    k.add_argument("peneliti", nargs="?", default=str(SHEET_PATH),
                   help=f"lembar peneliti (default {SHEET_PATH})")
    k.add_argument("pakar", nargs="?", default=str(AUDIT_PATH),
                   help=f"lembar audit yang sudah diisi pakar (default {AUDIT_PATH})")
    k.set_defaults(func=cmd_kappa)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()