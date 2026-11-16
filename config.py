# config.py
"""
Konfigurasi terpusat proyek Wazuh Alert Reduction (SOC INSTIKI).

Semua path, parameter pipeline, dan nama file artefak didefinisikan di sini.
Dipakai oleh: scripts/*, src/*, dan notebooks/*.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
DATA_RAW_DIR    = DATA_DIR / "raw"
DATA_PROC_DIR   = DATA_DIR / "processed"
MODELS_DIR      = BASE_DIR / "models"
REPORTS_DIR     = BASE_DIR / "reports"
FIGURES_DIR     = REPORTS_DIR / "figures"
API_DIR         = BASE_DIR / "api"

# ── Input Ekstraksi Dinamis ────────────────────────────────────────────
JSON_GZ_PATTERN   = "ossec-alerts-*.json.gz"
ARCHIVE_EXTENSION = ".tar.gz"

# ── Ekstraksi Field Nested JSON (Data Flattening) ──────────────────────
EXTRACT_FIELDS = {
    "timestamp":        ["timestamp"],
    "agent_name":       ["agent", "name"],
    "rule_level":       ["rule", "level"],
    "rule_description": ["rule", "description"],
    "rule_id":          ["rule", "id"],
    "rule_groups":      ["rule", "groups"],
    "decoder_name":     ["decoder", "name"],
    "location":         ["location"],
    "src_ip":           ["data", "srcip"],
    "url":              ["data", "url"],
    "http_status":      ["data", "id"],
}

# ── Feature Engineering Temporal (Data Preparation) ───────────────────
HOUR_GROUPS = {
    "dini_hari": range(0, 6),
    "pagi":      range(6, 12),
    "siang":     range(12, 18),
    "malam":     range(18, 24),
}

# ── Deduplikasi Temporal ──────────────────────────────────────────────
DEDUP_GROUP_COLS    = ["agent_name", "rule_id", "decoder_name", "rule_level", "location"]
TIME_WINDOW_MINUTES = 10
URL_MAX_CHARS       = 200

# ── K-Modes Clustering (Eksplorasi Struktur Alert) ─────────────────────
CLUSTER_FEATURES    = ["rule_level", "decoder_name", "location", "rule_groups"]
K_RANGE             = range(2, 11)
K_SEARCH_SAMPLE     = 20000
KMODES_INIT         = "Huang"
KMODES_N_INIT       = 10
KMODES_RANDOM_STATE = 42

# ── Supervised Classification (Random Forest & Triase SOC) ────────────
# Fitur X: murni atribut konteks (rule_id dan agent_name dikeluarkan)
CLASSIFICATION_FEATURES = ["rule_level", "decoder_name", "location", "rule_groups"]
NUMERIC_FEATURES        = ["rule_level"]
CATEGORICAL_FEATURES    = ["decoder_name", "location", "rule_groups"]

TARGET_LABEL_COL    = "label_final"
VALID_TRIAGE_LABELS = ["Dismiss", "Review", "Escalate"]

# Parameter Model
RF_RANDOM_STATE     = 42
RF_N_ESTIMATORS     = 200
RF_MAX_DEPTH        = 25
RF_CRITERION        = "gini"
RF_CLASS_WEIGHT     = "balanced"

# Ambang batas keyakinan (confidence) untuk usulan Dismiss pada rule baru
CONFIDENCE_THRESHOLD_DISMISS = 0.85

# ── Output & Artifact Files ────────────────────────────────────────────
PARQUET_CLEANED     = DATA_PROC_DIR / "cleaned.parquet"
PARQUET_DEDUPED     = DATA_PROC_DIR / "deduped.parquet"
PARQUET_CLUSTERED   = DATA_PROC_DIR / "clustered.parquet"
CSV_DEDUPED         = DATA_PROC_DIR / "deduped.csv"

# Tabel Kebijakan Rule (Hasil Anotasi Manusia)
CSV_RULE_LABELS     = DATA_PROC_DIR / "rule_labels_done.csv"
CSV_RULE_TEMPLATE   = DATA_PROC_DIR / "rule_labels.csv"

# Laporan & Evaluasi
CONFIG_RUN_JSON     = REPORTS_DIR / "run_config.json"
EVAL_CV_JSON        = REPORTS_DIR / "eval_rule_cross_validation.json"
CSV_CV_SUMMARY      = REPORTS_DIR / "cv_fold_summary.csv"

# Model Artifacts (.pkl)
MODEL_KMODES_FILE   = MODELS_DIR / "kmodes_wazuh.pkl"
MODEL_RF_FILE       = MODELS_DIR / "random_forest_wazuh.pkl"
ENCODER_FILE        = MODELS_DIR / "custom_label_encoders.pkl"
