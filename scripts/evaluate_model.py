# scripts/evaluate_model.py
"""
Script 4: Evaluasi Komprehensif Sistem Triase Alert Wazuh

Menyediakan dua mode evaluasi utama:
1. Mode 1: 5-Fold Stratified Cross-Validation Berbasis RULE (Simulasi Rule Baru di Latih).
2. Mode 2: Uji Lintas Periode (Out-of-Distribution Test) pada Bulan Lain (misal: Juni 2025).
   - Mengukur efisiensi Deduplikasi Temporal.
   - Mengukur Cakupan Kebijakan (Policy Table Coverage) dari 51 rule dikenal.
   - Menguji keandalan Model AI pada Rule Baru yang muncul di bulan tersebut.
   - Mengukur Reduksi Total Beban Analis.
"""
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from config import (
    DATA_RAW_DIR,
    PARQUET_DEDUPED,
    CSV_RULE_LABELS,
    CONFIG_RUN_JSON,
    TARGET_LABEL_COL,
    CLASSIFICATION_FEATURES,
    EVAL_CV_JSON,
    CSV_CV_SUMMARY,
    REPORTS_DIR,
    CONFIDENCE_THRESHOLD_DISMISS,
    TIME_WINDOW_MINUTES,
)
from src.transformers import (
    read_wazuh_logs,
    flatten_and_clean_logs,
    apply_feature_engineering,
    deduplicate,
    filter_period,
)
from src.labeling import load_rule_labels
from src.classification import (
    apply_rule_labels,
    fit_encoders,
    encode_features,
    encode_target,
    train_random_forest,
    predict_new_rules_triase,
    load_artifacts,
)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# MODE 1: 5-FOLD CROSS-VALIDATION BERBASIS RULE
# ═══════════════════════════════════════════════════════════════════════

def run_rule_level_cross_validation():
    log.info("=" * 70)
    log.info("MODE 1: 5-FOLD STRATIFIED CROSS-VALIDATION BERBASIS RULE")
    log.info("=" * 70)

    if not PARQUET_DEDUPED.exists():
        log.error(f"File {PARQUET_DEDUPED} belum ada. Jalankan scripts/prepare_data.py terlebih dahulu.")
        return

    df = pd.read_parquet(PARQUET_DEDUPED)
    df = apply_rule_labels(df)

    rule_table = (
        df.groupby("rule_id", as_index=False)
        .agg(
            label_final=(TARGET_LABEL_COL, "first"),
            total_alerts=("alert_count", "sum"),
            total_pola=("rule_id", "size"),
        )
    )

    log.info(f"Total rule unik untuk validasi silang: {len(rule_table)} rule")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    rules = rule_table["rule_id"].values
    rule_labels = rule_table["label_final"].values

    fold_results = []
    
    print("\n" + "=" * 70)
    print(f"{'Fold':<6} | {'Rule Latih':<12} | {'Rule Uji':<10} | {'Pola Uji':<10} | {'False Dismiss':<15} | {'Safe Reduksi (%)'}")
    print("=" * 70)

    for fold_idx, (train_rule_indices, test_rule_indices) in enumerate(skf.split(rules, rule_labels), 1):
        train_rules = set(rules[train_rule_indices])
        test_rules = set(rules[test_rule_indices])

        train_df = df[df["rule_id"].isin(train_rules)].copy()
        test_df = df[df["rule_id"].isin(test_rules)].copy()

        encoders = fit_encoders(train_df)
        X_train = encode_features(train_df, encoders)
        y_train = encode_target(train_df, encoders)
        X_test = encode_features(test_df, encoders)

        model = train_random_forest(X_train, y_train)
        y_pred_ai, confidences = predict_new_rules_triase(model, X_test, encoders)

        y_true_binary = np.where(test_df[TARGET_LABEL_COL] == "Dismiss", "Dismiss", "Attention")

        false_dismiss_ai = int(np.sum((y_true_binary == "Attention") & (y_pred_ai == "Dismiss")))
        pola_dismiss_ai = int(np.sum(y_pred_ai == "Dismiss"))
        safe_reduction_pola_ai = (pola_dismiss_ai / len(test_df)) * 100 if len(test_df) > 0 else 0

        alerts_total_test = test_df["alert_count"].sum()
        alerts_dismiss_ai = test_df.loc[y_pred_ai == "Dismiss", "alert_count"].sum()
        safe_reduction_alert_ai = (alerts_dismiss_ai / alerts_total_test) * 100 if alerts_total_test > 0 else 0

        acc_ai = accuracy_score(y_true_binary, y_pred_ai)
        f1_macro_ai = f1_score(y_true_binary, y_pred_ai, average="macro", zero_division=0)

        # Baseline 2 (rule_level <= 5)
        y_pred_base2 = np.where(test_df["rule_level"] <= 5, "Dismiss", "Attention")
        false_dismiss_base2 = int(np.sum((y_true_binary == "Attention") & (y_pred_base2 == "Dismiss")))
        pola_dismiss_base2 = int(np.sum(y_pred_base2 == "Dismiss"))
        safe_reduction_base2 = (pola_dismiss_base2 / len(test_df)) * 100 if len(test_df) > 0 else 0

        fold_stat = {
            "fold": fold_idx,
            "n_rule_train": len(train_rules),
            "n_rule_test": len(test_rules),
            "n_pola_test": len(test_df),
            "n_alerts_test": int(alerts_total_test),
            "rules_tested": sorted(list(test_rules)),
            "ai_false_dismissal": false_dismiss_ai,
            "ai_safe_reduction_pola_pct": round(safe_reduction_pola_ai, 2),
            "ai_safe_reduction_alert_pct": round(safe_reduction_alert_ai, 2),
            "ai_accuracy": round(acc_ai, 4),
            "ai_macro_f1": round(f1_macro_ai, 4),
            "baseline2_false_dismissal": false_dismiss_base2,
            "baseline2_safe_reduction_pct": round(safe_reduction_base2, 2),
        }
        fold_results.append(fold_stat)

        print(f"Fold {fold_idx:<2} | {len(train_rules):<12} | {len(test_rules):<10} | {len(test_df):<10,} | {false_dismiss_ai:<15} | {safe_reduction_alert_ai:5.2f}% (AI)")
        print(f"       | Baseline 2 (Level <= 5)                      | {false_dismiss_base2:<15} | {safe_reduction_base2:5.2f}%")
        print("-" * 70)

    df_results = pd.DataFrame(fold_results)
    df_results.to_csv(CSV_CV_SUMMARY, index=False)

    avg_false_dismiss_ai = df_results["ai_false_dismissal"].mean()
    avg_reduction_alert_ai = df_results["ai_safe_reduction_alert_pct"].mean()
    avg_macro_f1_ai = df_results["ai_macro_f1"].mean()
    avg_false_dismiss_base2 = df_results["baseline2_false_dismissal"].mean()
    avg_reduction_base2 = df_results["baseline2_safe_reduction_pct"].mean()

    summary_json = {
        "metodologi": "5-Fold Stratified Cross-Validation on Rules (Unknown Rule Simulation)",
        "fitur_x": CLASSIFICATION_FEATURES,
        "ambang_confidence_dismiss": CONFIDENCE_THRESHOLD_DISMISS,
        "total_rule": len(rule_table),
        "rata_rata_ai": {
            "rata_rata_false_dismissal_per_fold": round(avg_false_dismiss_ai, 2),
            "rata_rata_safe_reduction_alert_pct": round(avg_reduction_alert_ai, 2),
            "rata_rata_macro_f1": round(avg_macro_f1_ai, 4),
        },
        "rata_rata_baseline_heuristik_level": {
            "rata_rata_false_dismissal_per_fold": round(avg_false_dismiss_base2, 2),
            "rata_rata_safe_reduction_pct": round(avg_reduction_base2, 2),
        },
        "detail_per_fold": fold_results,
    }

    with open(EVAL_CV_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("KESIMPULAN UJI TANDING 5-FOLD CROSS-VALIDATION:")
    print("=" * 70)
    print(f"1. Model Random Forest Kontekstual:")
    print(f"   * Rata-rata False Dismissal (Bahaya Lolos) : {avg_false_dismiss_ai:.2f} pola")
    print(f"   * Rata-rata Reduksi Aman Alert            : {avg_reduction_alert_ai:.2f}%")
    print(f"   * Rata-rata Macro-F1                      : {avg_macro_f1_ai:.4f}")
    print()
    print(f"2. Baseline Heuristik (Rule Level <= 5):")
    print(f"   * Rata-rata False Dismissal (Bahaya Lolos) : {avg_false_dismiss_base2:.2f} pola")
    print(f"   * Rata-rata Reduksi Alert                 : {avg_reduction_base2:.2f}%")
    print("=" * 70)
    print(f"Laporan lengkap disimpan di: {EVAL_CV_JSON}")
    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════
# MODE 2: UJI LINTAS PERIODE (OUT-OF-DISTRIBUTION TEST)
# ═══════════════════════════════════════════════════════════════════════

def run_cross_period_evaluation():
    log.info("=" * 70)
    log.info("MODE 2: UJI LINTAS PERIODE (OUT-OF-DISTRIBUTION TEST)")
    log.info("=" * 70)

    # 1. Baca Konfigurasi Run Latih (untuk sinkronisasi agen blacklist)
    blacklist = ["DVWA", "dfir-iris", "proxy-manager", "soc-1"]
    if CONFIG_RUN_JSON.exists():
        try:
            with open(CONFIG_RUN_JSON, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                blacklist = cfg.get("agen_diblacklist", blacklist)
        except Exception:
            pass

    log.info(f"Agen yang disaring (blacklist): {blacklist}")

    # 2. Muat Model Produksi & Tabel Kebijakan
    try:
        model, encoders = load_artifacts()
    except FileNotFoundError:
        log.error("Model belum dilatih! Jalankan 'python -m scripts.train_model' terlebih dahulu.")
        return

    policy_map = load_rule_labels()
    known_rules = set(policy_map.keys())
    log.info(f"Tabel kebijakan memuat {len(known_rules)} rule dikenal.")

    # 3. Minta Periode Uji
    periode_input = input("\nMasukkan periode yang ingin diuji (Contoh: '2025-06' atau '2026-02'): ").strip()
    if not periode_input:
        periode_input = "2025-06"
        print(f"Menggunakan periode default: {periode_input}")

    # 4. Ekstraksi dan Pembersihan Log Periode Uji
    log.info(f"Membaca data mentah untuk periode {periode_input}...")
    raw_records = read_wazuh_logs(DATA_RAW_DIR)
    df_raw = flatten_and_clean_logs(raw_records)
    df_period = filter_period(df_raw, periode_input, None)

    if df_period.empty:
        log.error(f"Tidak ada data log pada periode '{periode_input}'.")
        return

    # Saring agen blacklist
    df_filtered = df_period[~df_period["agent_name"].isin(blacklist)].copy()
    total_raw_alerts = len(df_filtered)
    log.info(f"Total alert mentah (setelah filter agen): {total_raw_alerts:,} baris")

    # 5. Deduplikasi Temporal (10 Menit)
    log.info("Menjalankan deduplikasi temporal...")
    df_fe = apply_feature_engineering(df_filtered)
    deduped = deduplicate(df_fe)
    total_dedup_patterns = len(deduped)

    dedup_reduction_pct = (1 - (total_dedup_patterns / total_raw_alerts)) * 100
    log.info(f"Alert mentah : {total_raw_alerts:,} -> Pola unik dedup: {total_dedup_patterns:,}")
    log.info(f"Reduksi dari Deduplikasi: {dedup_reduction_pct:.2f}%")

    # 6. Pemisahan: Rule Dikenal vs Rule Baru
    deduped["is_known"] = deduped["rule_id"].astype(str).isin(known_rules)
    df_known = deduped[deduped["is_known"]].copy()
    df_unknown = deduped[~deduped["is_known"]].copy()

    known_alerts = df_known["alert_count"].sum()
    unknown_alerts = df_unknown["alert_count"].sum() if not df_unknown.empty else 0

    coverage_pct = (known_alerts / total_raw_alerts) * 100
    log.info("\n" + "-" * 60)
    log.info("CAKUPAN TABEL KEBIJAKAN (POLICY TABLE COVERAGE):")
    log.info(f"  * Pola Rule Dikenal : {len(df_known):,} pola ({known_alerts:,} alerts | {coverage_pct:.2f}%)")
    log.info(f"  * Pola Rule Baru    : {len(df_unknown):,} pola ({unknown_alerts:,} alerts | {100-coverage_pct:.2f}%)")
    log.info("-" * 60)

    # 7. Triase Deterministik pada Rule Dikenal
    df_known["triase_decision"] = df_known["rule_id"].astype(str).map(policy_map)
    alerts_dismiss_known = int(df_known.loc[df_known["triase_decision"] == "Dismiss", "alert_count"].sum())
    patterns_dismiss_known = int((df_known["triase_decision"] == "Dismiss").sum())

    # 8. Inferensi Model AI pada Rule Baru
    alerts_dismiss_unknown = 0
    patterns_dismiss_unknown = 0
    if not df_unknown.empty:
        X_unk = encode_features(df_unknown, encoders)
        y_pred_unk, _ = predict_new_rules_triase(model, X_unk, encoders)
        df_unknown["triase_decision"] = y_pred_unk
        alerts_dismiss_unknown = int(df_unknown.loc[df_unknown["triase_decision"] == "Dismiss", "alert_count"].sum())
        patterns_dismiss_unknown = int((df_unknown["triase_decision"] == "Dismiss").sum())
        log.info(f"Rule Baru yang diusulkan Dismiss oleh AI: {alerts_dismiss_unknown:,} alerts ({patterns_dismiss_unknown:,} pola)")
        log.info(f"Rule Baru yang diarahkan Attention     : {unknown_alerts - alerts_dismiss_unknown:,} alerts ({len(df_unknown) - patterns_dismiss_unknown:,} pola)")
    else:
        log.info("Tidak ada rule baru yang terdeteksi.")

    # 9. Kalkulasi Dual-Perspektif: Alert Mentah vs Pola Tiket Analis
    # PERSPEKTIF A: Alert Mentah (Volume Log)
    total_dismiss_alerts = alerts_dismiss_known + alerts_dismiss_unknown
    alerts_for_analyst = total_raw_alerts - total_dismiss_alerts
    raw_dismiss_reduction_pct = (total_dismiss_alerts / total_raw_alerts) * 100

    # PERSPEKTIF B: Pola Deduplikasi (Tiket Dashboard Analis)
    total_dismiss_patterns = patterns_dismiss_known + patterns_dismiss_unknown
    patterns_for_analyst = total_dedup_patterns - total_dismiss_patterns
    pattern_dismiss_reduction_pct = (total_dismiss_patterns / total_dedup_patterns) * 100 if total_dedup_patterns > 0 else 0
    
    # Reduksi Beban Total terhadap Beban Awal (dari Raw Alert -> Pola yang harus diperiksa analis)
    total_effective_workload_reduction_pct = (1 - (patterns_for_analyst / total_raw_alerts)) * 100

    # Estimasi Beban Kerja Operasional Harian & Per Jam (Asumsi 30 Hari per Bulan)
    raw_alerts_per_day = round(total_raw_alerts / 30, 1)
    raw_alerts_per_hour = round(total_raw_alerts / (30 * 24), 1)
    patterns_per_day = round(patterns_for_analyst / 30, 1)
    patterns_per_hour = round(patterns_for_analyst / (30 * 24), 2)

    print("\n" + "=" * 76)
    print(f"HASIL EVALUASI DUAL-PERSPEKTIF LINTAS PERIODE (Bulan: {periode_input})")
    print("=" * 76)
    print("1. PERSPEKTIF VOLUME ALERT MENTAH:")
    print(f"   * Total Alert Mentah Asli              : {total_raw_alerts:,} baris")
    print(f"   * Cakupan Kebijakan (51 Rule)          : {coverage_pct:.2f}% dari seluruh alert")
    print(f"   * Alert Diredam Otomatis (Dismiss)     : {total_dismiss_alerts:,} baris ({raw_dismiss_reduction_pct:.2f}%)")
    print(f"     - Dari Tabel Kebijakan Dikenal       : {alerts_dismiss_known:,} baris")
    print(f"     - Dari Usulan Model AI (Rule Baru)   : {alerts_dismiss_unknown:,} baris")
    print(f"   * Alert Butuh Penanganan (Review/Esc)  : {alerts_for_analyst:,} baris")
    print()
    print("2. PERSPEKTIF POLA DEDUPLIKASI (TIKET DASHBOARD ANALIS):")
    print(f"   * Total Pola Dedup (10 Menit)          : {total_dedup_patterns:,} pola (Reduksi dedup: {dedup_reduction_pct:.2f}%)")
    print(f"   * Pola Diredam Otomatis (Dismiss)      : {total_dismiss_patterns:,} pola ({pattern_dismiss_reduction_pct:.2f}%)")
    print(f"     - Dari Tabel Kebijakan Dikenal       : {patterns_dismiss_known:,} pola")
    print(f"     - Dari Usulan Model AI (Rule Baru)   : {patterns_dismiss_unknown:,} pola")
    print(f"   * Pola Tiket Tersisa untuk Analis SOC  : {patterns_for_analyst:,} pola")
    print(f"   * TOTAL REDUKSI BEBAN TIKET ANALIS     : {total_effective_workload_reduction_pct:.2f}%")
    print()
    print("3. DAMPAK TERHADAP ALERT FATIGUE (ASUMSI 30 HARI):")
    print(f"   * Tanpa Sistem (Raw Alert)             : ~{raw_alerts_per_day:,.0f} alert/hari  |  ~{raw_alerts_per_hour:,.1f} alert/jam")
    print(f"   * Dengan Sistem Kami (Pola Tiket)      : ~{patterns_per_day:,.0f} tiket/hari  |  ~{patterns_per_hour:,.1f} tiket/jam")
    print("=" * 76)

    # Simpan Hasil ke JSON
    out_file = REPORTS_DIR / f"eval_lintas_periode_{periode_input}.json"
    result_data = {
        "periode_uji": periode_input,
        "perspektif_alert_mentah": {
            "total_alert_mentah": int(total_raw_alerts),
            "cakupan_kebijakan_persen": round(coverage_pct, 2),
            "alert_dikenal": int(known_alerts),
            "alert_baru": int(unknown_alerts),
            "alert_diredam_dismiss": int(total_dismiss_alerts),
            "alert_diredam_kebijakan": int(alerts_dismiss_known),
            "alert_diredam_ai": int(alerts_dismiss_unknown),
            "alert_tersisa_analis": int(alerts_for_analyst),
            "persentase_reduksi_alert_mentah": round(raw_dismiss_reduction_pct, 2),
        },
        "perspektif_pola_deduplikasi": {
            "total_pola_dedup": int(total_dedup_patterns),
            "efisiensi_dedup_persen": round(dedup_reduction_pct, 2),
            "pola_diredam_dismiss": int(total_dismiss_patterns),
            "pola_diredam_kebijakan": int(patterns_dismiss_known),
            "pola_diredam_ai": int(patterns_dismiss_unknown),
            "pola_tersisa_analis": int(patterns_for_analyst),
            "persentase_reduksi_pola": round(pattern_dismiss_reduction_pct, 2),
            "persentase_reduksi_beban_kerja_total": round(total_effective_workload_reduction_pct, 2),
        },
        "analisis_alert_fatigue_operasional": {
            "beban_awal_per_hari": raw_alerts_per_day,
            "beban_awal_per_jam": raw_alerts_per_hour,
            "beban_sistem_per_hari": patterns_per_day,
            "beban_sistem_per_jam": patterns_per_hour,
        },
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    log.info(f"Laporan evaluasi disimpan di: {out_file}\n")


def main():
    print("\n" + "=" * 55)
    print("PILIH MODE EVALUASI:")
    print("=" * 55)
    print("  [1] 5-Fold Cross-Validation Berbasis Rule (Simulasi Rule Baru)")
    print("  [2] Uji Lintas Periode / Bulan Lain (Out-of-Distribution)")
    print("=" * 55)
    pilihan = input("Masukkan pilihan (1 / 2): ").strip()

    if pilihan == "2":
        run_cross_period_evaluation()
    else:
        run_rule_level_cross_validation()


if __name__ == "__main__":
    main()
