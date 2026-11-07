# pipeline_c_evaluate.py
"""
Script C: Evaluasi Model pada Periode Uji
Mengacu pada arsitektur CRISP-DM Fase 5 (Evaluation) untuk Log Wazuh SOC INSTIKI.

Alur:
1. Baca konfigurasi run latih (reports/run_config.json): blacklist agen dan periode latih.
2. Pilih periode uji (tahun / bulan / tanggal), di luar periode latih.
3. Baca log, bersihkan, filter periode & blacklist, dedup 10 menit.
4. Beri label uji: dari rule_labels.csv; rule baru memakai usulan otomatis dari grup Wazuh.
5. Prediksi dengan model terlatih, lalu evaluasi: total, rule dikenal vs rule baru, FPR,
   recall per kelas, unknown rate, metrik tambahan (macro-F1, balanced accuracy, MCC,
   akurasi berbobot alert_count), tabel kesalahan per rule, dan kurva abstain.

Keluaran utama:
- reports/eval_<awal>_<akhir>.json, reports/figures/confusion_matrix_uji_<...>.png

Catatan:
- Label uji untuk rule baru belum divalidasi pakar (dicatat di keluaran).
- Pada sistem yang meredam alert, kesalahan paling berbahaya adalah serangan yang salah
  dilabeli rutin. Perhatikan recall kelas Web_Attack.
- Model HARUS dilatih ulang (Script B) setelah perubahan fitur di config.py.
"""
import json
import logging

import matplotlib
matplotlib.use("Agg")  # agar plt.show() tidak memblokir script

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
)

from config import (
    DATA_RAW_DIR, REPORTS_DIR, CONFIG_RUN_JSON, TARGET_LABEL_COL,
)
from src.transformers import (
    read_wazuh_logs, flatten_and_clean_logs, apply_feature_engineering,
    deduplicate, filter_period, period_bounds,
)
from src.classification import (
    load_artifacts, encode_features, encode_target, evaluate_model,
    unknown_flags, abstain_curve,
)
from src.labeling import load_rule_labels, suggest_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def prompt_test_period():
    print("\nFormat: 2026 | 2026-03 | 2026-03-15")
    while True:
        start = input("Periode UJI - awal (wajib)   : ").strip()
        end = input("Periode UJI - akhir (ENTER = sama dengan awal): ").strip() or None
        if not start:
            print("Periode awal wajib diisi.")
            continue
        try:
            period_bounds(start)
            if end:
                period_bounds(end)
        except ValueError:
            print("Format tidak valid, coba lagi.")
            continue
        return start, end


def summarize_subset(mask, model, X, y, class_names) -> dict:
    """Accuracy dan recall per kelas untuk satu subset baris uji."""
    if int(mask.sum()) == 0:
        return {"n_pola": 0}
    pred = model.predict(X[mask])
    labels = list(range(len(class_names)))
    rep = classification_report(y[mask], pred, labels=labels, target_names=class_names,
                                output_dict=True, zero_division=0)
    return {
        "n_pola": int(mask.sum()),
        "accuracy": round(float(accuracy_score(y[mask], pred)), 4),
        "recall_per_kelas": {c: round(rep[c]["recall"], 4) for c in class_names if rep[c]["support"] > 0},
        "jumlah_per_kelas": {c: int(rep[c]["support"]) for c in class_names if rep[c]["support"] > 0},
    }


def run():
    with open(CONFIG_RUN_JSON, encoding="utf-8") as f:
        run_cfg = json.load(f)
    blacklist = run_cfg.get("agen_diblacklist", [])
    train_p = run_cfg.get("periode_latih", {})
    log.info(f"Blacklist dari run latih: {blacklist}")
    log.info(f"Periode latih: {train_p}")

    model, encoders = load_artifacts()
    rule_labels = load_rule_labels()
    class_names = encoders[TARGET_LABEL_COL].classes_

    start, end = prompt_test_period()

    if train_p.get("awal"):
        t_lo = period_bounds(train_p["awal"])[0]
        t_hi = period_bounds(train_p.get("akhir") or train_p["awal"])[1]
        s_lo = period_bounds(start)[0]
        s_hi = period_bounds(end or start)[1]
        if s_lo < t_hi and t_lo < s_hi:
            log.warning("Periode uji BERIRISAN dengan periode latih, hasil bisa terlalu optimis.")

    records = read_wazuh_logs(DATA_RAW_DIR)
    df = flatten_and_clean_logs(records)
    df = filter_period(df, start, end)
    df = df[~df["agent_name"].isin(blacklist)]
    if df.empty:
        log.error("Tidak ada data uji setelah filter periode & blacklist.")
        return
    log.info(f"Data uji: {len(df):,} log mentah")

    deduped = deduplicate(apply_feature_engineering(df))
    log.info(f"Pola uji setelah dedup: {len(deduped):,}")
    log.info(f"Reduksi dedup: {(1 - len(deduped) / len(df)) * 100:.2f}%")

    # Label uji: rule_labels.csv bila rule sudah ada; bila rule baru -> usulan otomatis dari grup Wazuh
    rid = deduped["rule_id"].astype(str)
    in_catalog = rid.isin(rule_labels.keys())
    deduped[TARGET_LABEL_COL] = [
        rule_labels[r] if r in rule_labels else suggest_label(r, g)
        for r, g in zip(rid, deduped["rule_groups"])
    ]

    X = encode_features(deduped, encoders)
    y = encode_target(deduped, encoders)

    tag = f"{start}_{end or start}"
    res = evaluate_model(model, X, y, encoders, fig_name=f"confusion_matrix_uji_{tag}.png")

    # Metrik tambahan, termasuk akurasi berbobot jumlah alert (beban SOC sebenarnya)
    pred = model.predict(X)
    w = deduped["alert_count"].to_numpy()
    present = np.unique(np.concatenate([y, pred]))
    present = present[present >= 0]
    extra = {
        "macro_f1": round(float(f1_score(y, pred, labels=present, average="macro", zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "mcc": round(float(matthews_corrcoef(y, pred)), 4),
        "accuracy_berbobot_alert": round(float(accuracy_score(y, pred, sample_weight=w)), 4),
    }
    print("\nMetrik tambahan:", json.dumps(extra, indent=2))

    # Rule dikenal vs rule baru (belum ada saat pelatihan)
    known = rid.isin(encoders["rule_id"].classes_).to_numpy()
    sub_known = summarize_subset(known, model, X, y, class_names)
    sub_new = summarize_subset(~known, model, X, y, class_names)
    print("\nRule DIKENAL saat pelatihan :", json.dumps(sub_known, indent=2))
    print("\nRule BARU (belum dikenal)   :", json.dumps(sub_new, indent=2))
    if sub_new["n_pola"]:
        print("\nRule baru yang muncul:", sorted(rid[~known].unique().tolist()))

    # Rule mana yang paling sering salah diprediksi (untuk ditinjau manual)
    tgt = encoders[TARGET_LABEL_COL]
    wrong = pred != y
    salah = pd.DataFrame()
    if wrong.any():
        bad = deduped.loc[wrong, ["rule_id", "alert_count"]].copy()
        bad["label_benar"] = tgt.inverse_transform(y[wrong])
        bad["prediksi"] = tgt.inverse_transform(pred[wrong])
        salah = (bad.groupby(["rule_id", "label_benar", "prediksi"])
                    .agg(pola=("alert_count", "size"), alert=("alert_count", "sum"))
                    .sort_values("pola", ascending=False).head(30).reset_index())
        print("\nRule yang paling sering salah diprediksi:")
        print(salah.to_string(index=False))

    # Kurva abstain: berapa persen pola yang bisa diputuskan model pada ambang tertentu
    curve = abstain_curve(model, X, y)
    print("\nKurva abstain (model ragu -> diserahkan ke analis):")
    print(curve.to_string(index=False))

    # Unknown rate: hanya fitur kategorikal (rule_level numerik tidak bisa 'asing')
    flags = unknown_flags(deduped, encoders)
    any_unk = np.column_stack(list(flags.values())).any(axis=1)
    unknown = {
        "baris_dengan_nilai_asing_%": round(float(any_unk.mean() * 100), 2),
        "berbobot_alert_count_%": round(float(np.average(any_unk, weights=w) * 100), 2),
        "per_fitur_%": {c: round(float(f.mean() * 100), 2) for c, f in flags.items()},
    }
    print("\nUnknown rate (nilai tidak dikenal saat inferensi):")
    print(json.dumps(unknown, indent=2))

    out = {
        "periode_uji": {"awal": start, "akhir": end or start},
        "periode_latih": train_p,
        "log_mentah": len(df),
        "pola_dedup": len(deduped),
        "reduksi_dedup_%": round((1 - len(deduped) / len(df)) * 100, 2),
        "distribusi_label_pola": deduped[TARGET_LABEL_COL].value_counts().to_dict(),
        "pola_label_dari_katalog_%": round(float(in_catalog.mean() * 100), 2),
        "accuracy_total": res["accuracy"],
        "metrik_tambahan": extra,
        "fpr": res["fpr"],
        "rule_dikenal": sub_known,
        "rule_baru": sub_new,
        "kesalahan_terbanyak": salah.to_dict("records"),
        "kurva_abstain": curve.to_dict("records"),
        "unknown_rate": unknown,
        "sumber_label_uji": "rule_labels.csv; rule baru memakai usulan otomatis dari grup Wazuh (belum divalidasi pakar)",
    }
    path = REPORTS_DIR / f"eval_{tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Hasil evaluasi disimpan di: {path}")


if __name__ == "__main__":
    run()