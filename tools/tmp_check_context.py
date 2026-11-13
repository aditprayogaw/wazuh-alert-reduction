# tmp_check_context.py  (sementara, boleh dihapus setelah dipakai)
"""
Cek konteks web untuk satu periode: variasi per pola, lalu statistik per IP.
Hanya membaca dan mencetak, tidak menulis file apa pun. Alamat IP tidak ditampilkan.

Letakkan di folder utama proyek (sejajar config.py), lalu jalankan:
    python tmp_check_context.py 2025-06
"""
import sys

import pandas as pd

from config import DATA_RAW_DIR
from src.transformers import (read_wazuh_logs, flatten_and_clean_logs, filter_period,
                              apply_feature_engineering, deduplicate)

period = sys.argv[1] if len(sys.argv) > 1 else "2025-06"
df = filter_period(flatten_and_clean_logs(read_wazuh_logs(DATA_RAW_DIR)), period)
d = deduplicate(apply_feature_engineering(df))
w = d[d.n_url > 0]
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 45)

print(f"=== BAGIAN 1: VARIASI PER POLA ({period}) ===")
print(f"{len(df):,} log mentah -> {len(d):,} pola; pola web: {len(w):,}")
print(w.groupby("rule_id").agg(
    pola=("alert_count", "size"), alert=("alert_count", "sum"),
    ip_med=("n_src_ip", "median"), ip_max=("n_src_ip", "max"),
    url_med=("n_url", "median"), top_med=("top_ip_share", "median"),
    s4=("share_4xx", "mean"), desc=("rule_description", "first"),
).sort_values("alert", ascending=False).head(15).round(2).to_string())

print("\n=== BAGIAN 2: STATISTIK PER IP (alamat IP tidak ditampilkan) ===")
web = df[df["url"].notna()].copy()
total = len(web)
by_ip = web.groupby("src_ip").agg(
    alert=("rule_id", "size"),
    aturan=("rule_id", "nunique"),
    url=("url", "nunique"),
    hari=("timestamp", lambda t: t.dt.date.nunique()),
    sukses=("http_status", lambda s: int(((s >= 200) & (s < 300)).sum())),
).sort_values("alert", ascending=False)
print(f"IP berbeda pada alert web: {len(by_ip):,}")
for k in (1, 5, 10, 50):
    print(f"  {k:>3} IP teratas = {by_ip['alert'].head(k).sum() / total:.1%} dari alert web")
print("Ringkasan per IP:")
print(by_ip.reset_index(drop=True).describe(percentiles=[.5, .9])
      .loc[["count", "50%", "90%", "max"]].round(1).to_string())
n31101 = web[web.rule_id == "31101"]["src_ip"].nunique()
n31151 = web[web.rule_id == "31151"]["src_ip"].nunique()
print(f"IP berbeda: 31101 = {n31101}, 31151 = {n31151}")
multi = web.groupby("src_ip")["rule_id"].nunique()
print(f"IP yang memicu lebih dari 1 jenis rule: {(multi > 1).sum()} dari {len(multi)}")