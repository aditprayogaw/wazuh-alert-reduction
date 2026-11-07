# pipeline_e_annotation.py
"""
Script E: Lembar anotasi untuk pelabelan manual + penghitung kesepakatan (kappa).

Tujuan: membuat sampel pola yang dilabeli manual oleh dua orang (Abaikan / Tinjau / Eskalasi),
lalu mengukur seberapa sering keduanya sepakat.

Pemakaian:
  1) Buat lembar anotasi (setelah Script A):
       python pipeline_e_annotation.py sheet --n 300 --top 80 --pilot 40
     -> data/processed/anotasi/anotasi_A.csv, anotasi_B.csv  (untuk pelabel, tanpa petunjuk label)
        data/processed/anotasi/anotasi_kunci.csv              (untuk peneliti, jangan dibagikan)
  2) Kedua pelabel mengisi kolom `label` (dan `alasan` bila perlu) di salinannya masing-masing.
  3) Hitung kesepakatan dan daftar selisih untuk ditengahi:
       python pipeline_e_annotation.py kappa anotasi_A.csv anotasi_B.csv

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
VALID = ["Abaikan", "Tinjau", "Eskalasi"]
EVIDENCE_COLS = ["pola_id", "rule_id", "rule_description", "rule_level", "agent_name",
                 "location", "decoder_name", "hour_group", "alert_count"]
OPTIONAL_COLS = ["timestamp", "first_seen", "last_seen"]
RISKY_LEVEL = 10


def cmd_sheet(args):
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
    annot = sample[cols].copy()
    annot["label"] = ""
    annot["alasan"] = ""

    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("A", "B"):
        annot.to_csv(ANNOT_DIR / f"anotasi_{name}.csv", index=False, encoding="utf-8-sig")
    key_cols = [c for c in ["pola_id", "strata", "kategori", "alert_count", "rule_groups"] if c in sample.columns]
    sample[key_cols].to_csv(ANNOT_DIR / "anotasi_kunci.csv", index=False, encoding="utf-8-sig")

    share = sample["alert_count"].sum() / df["alert_count"].sum() * 100
    print(f"Sampel: {len(sample)} pola (pilot: {int(sample['pilot'].sum())})")
    print(sample["strata"].value_counts().to_string())
    print(f"Sampel mewakili {share:.1f}% dari total alert mentah pada data ini.")
    print(f"File ada di: {ANNOT_DIR}")


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


def _read_annot(path, keep):
    d = pd.read_csv(path, dtype={"rule_id": str}, encoding="utf-8-sig").fillna("")
    d["label"] = d["label"].astype(str).str.strip().str.title()
    return d[[c for c in keep if c in d.columns]]


def _report(m: pd.DataFrame, judul: str):
    k = cohen_kappa_score(m["label_A"], m["label_B"], labels=VALID)
    kw = cohen_kappa_score(m["label_A"], m["label_B"], labels=VALID, weights="linear")
    agree = (m["label_A"] == m["label_B"]).mean() * 100
    print(f"\n== {judul} (n={len(m)}) ==")
    print(f"Persen sepakat       : {agree:.1f}%")
    print(f"Cohen's kappa        : {k:.3f} ({landis_koch(k)})")
    print(f"Kappa berbobot linier: {kw:.3f} (selisih Abaikan-Eskalasi dihitung lebih berat)")
    print(pd.crosstab(m["label_A"], m["label_B"]).reindex(index=VALID, columns=VALID, fill_value=0))


def cmd_kappa(args):
    a = _read_annot(args.file_a, ["pola_id", "pilot", "rule_id", "rule_description",
                                  "alert_count", "label", "alasan"])
    b = _read_annot(args.file_b, ["pola_id", "label", "alasan"])
    m = a.merge(b, on="pola_id", suffixes=("_A", "_B"))

    invalid = m[~m["label_A"].isin(VALID) | ~m["label_B"].isin(VALID)]
    if len(invalid):
        print(f"PERINGATAN: {len(invalid)} baris dilewati (label kosong/tidak valid). "
              f"Label yang sah: {VALID}")
    m = m.drop(invalid.index)
    if m.empty:
        print("Tidak ada baris yang bisa dihitung.")
        return

    if "pilot" in m.columns and (m["pilot"].astype(str) == "1").any():
        _report(m[m["pilot"].astype(str) == "1"], "Pilot saja")
    _report(m, "Semua baris")

    diff = m[m["label_A"] != m["label_B"]].copy()
    diff["label_final"] = ""
    cols = [c for c in ["pola_id", "rule_id", "rule_description", "alert_count",
                        "label_A", "label_B", "alasan_A", "alasan_B", "label_final"] if c in diff.columns]
    out = ANNOT_DIR / "selisih_untuk_adjudikasi.csv"
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    diff[cols].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n{len(diff)} selisih disimpan di: {out} (isi kolom label_final setelah ditengahi)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sheet", help="buat lembar anotasi")
    s.add_argument("--n", type=int, default=300, help="target jumlah pola (default 300)")
    s.add_argument("--top", type=int, default=80, help="jumlah pola terberat (default 80)")
    s.add_argument("--per-risky", dest="per_risky", type=int, default=10,
                   help="maks pola per rule berisiko (default 10)")
    s.add_argument("--pilot", type=int, default=40, help="jumlah pola pilot (default 40)")
    s.set_defaults(func=cmd_sheet)

    k = sub.add_parser("kappa", help="hitung kesepakatan dua pelabel")
    k.add_argument("file_a")
    k.add_argument("file_b")
    k.set_defaults(func=cmd_kappa)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()