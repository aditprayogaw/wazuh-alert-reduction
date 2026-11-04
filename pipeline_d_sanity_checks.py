# pipeline_d_sanity_checks.py
"""
Script D: Uji validitas model (bukan bagian alur produksi).

Tujuan: membuktikan bahwa hasil Script B/C tidak semu (tidak bocor / tidak sekadar menghafal).
Prasyarat: CLASSIFICATION_FEATURES sudah ada di config.py dan src/classification.py
           sudah memakainya (tanpa rule_id sebagai fitur).
Jalankan setelah Script A dan B:  python pipeline_d_sanity_checks.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from config import (
    PARQUET_CLUSTERED, TARGET_LABEL_COL as LABEL,
    CLASSIFICATION_FEATURES, RF_RANDOM_STATE,
)
from src.classification import (
    apply_rule_labels, split_data, fit_encoders, encode_features,
    encode_target, train_random_forest,
)

df = apply_rule_labels(pd.read_parquet(PARQUET_CLUSTERED))
df["rule_id"] = df["rule_id"].astype(str)

# ── [0] Label murni fungsi rule_id? Dan apakah fitur lain menjadi "proxy" rule_id? ──
print("[0] Struktur label")
per_rule = df.groupby("rule_id")[LABEL].nunique()
print(f"  Jumlah rule unik: {len(per_rule)} | maks label berbeda per rule: {per_rule.max()}")
proxy_cols = [c for c in CLASSIFICATION_FEATURES if c != "hour_group"]
combo = df.groupby(proxy_cols, dropna=False)
n_rule = combo["rule_id"].nunique()
print(f"  Kombinasi {proxy_cols}: {len(n_rule)} unik, {(n_rule == 1).mean()*100:.1f}% menunjuk tepat 1 rule")
pure = combo[LABEL].transform("nunique") == 1
print(f"  Baris yang labelnya sudah pasti dari kombinasi fitur saja: {pure.mean()*100:.1f}%")

# ── Split baris acak (sama seperti Script B) ──
train_df, test_df = split_data(df)
enc = fit_encoders(train_df)
Xtr, ytr = encode_features(train_df, enc), encode_target(train_df, enc)
Xte, yte = encode_features(test_df, enc), encode_target(test_df, enc)
model = train_random_forest(Xtr, ytr)
pred = model.predict(Xte)

# ── [1] Feature importance ──
print("\n[1] Feature importance")
print(pd.Series(model.feature_importances_, index=CLASSIFICATION_FEATURES)
      .sort_values(ascending=False).round(4).to_string())

# ── [2] Baseline tanpa ML: lookup rule_id -> label ──
lookup = train_df.groupby("rule_id")[LABEL].agg(lambda s: s.mode().iloc[0])
pl = test_df["rule_id"].map(lookup)
ok = pl.notna()
print(f"\n[2] Baseline lookup rule_id: akurasi {(pl[ok] == test_df.loc[ok, LABEL]).mean():.4f} "
      f"pada {ok.mean()*100:.1f}% baris uji")
print(f"    Model RF (split baris acak): akurasi {accuracy_score(yte, pred):.4f}, "
      f"macro-F1 {f1_score(yte, pred, average='macro', zero_division=0):.4f}")

# ── [3] Duplikasi vektor fitur antara latih dan uji ──
tr_set = set(map(tuple, train_df[CLASSIFICATION_FEATURES].astype(str).values))
te_tup = list(map(tuple, test_df[CLASSIFICATION_FEATURES].astype(str).values))
print(f"\n[3] Baris uji yang vektor fiturnya sudah ada di data latih: "
      f"{100 * np.mean([t in tr_set for t in te_tup]):.1f}%")

# ── [4] Generalisasi ke RULE BARU: rule di data uji tidak ada di data latih ──
print("\n[4] StratifiedGroupKFold per rule_id (simulasi rule baru)")
sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RF_RANDOM_STATE)
for i, (tr, te) in enumerate(sgkf.split(df, df[LABEL], groups=df["rule_id"]), 1):
    a, b = df.iloc[tr], df.iloc[te]
    e = fit_encoders(a)
    m = train_random_forest(encode_features(a, e), encode_target(a, e))
    p = np.array(e[LABEL].inverse_transform(m.predict(encode_features(b, e))))
    y = b[LABEL].to_numpy()
    wa = y == "Web_Attack"
    wa_rec = (p[wa] == "Web_Attack").mean() if wa.any() else float("nan")
    wa_as_err = (p[wa] == "Web_Error").mean() if wa.any() else float("nan")
    print(f"  fold {i}: rule uji={b['rule_id'].nunique():>2} | acc={accuracy_score(y, p):.3f} | "
          f"macro-F1={f1_score(y, p, average='macro', zero_division=0):.3f} | "
          f"recall Web_Attack={wa_rec:.3f} (salah jadi Web_Error: {wa_as_err:.3f})")