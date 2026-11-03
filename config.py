# config.py
"""
Konfigurasi terpusat proyek Wazuh Alert Reduction (SOC INSTIKI).

Semua path, parameter pipeline, dan nama file artefak didefinisikan di sini.
Dipakai oleh: pipeline_cron.py (Script A), pipeline_b_train_model.py (Script B),
pipeline_c_evaluate.py (Script C), src/*, dan notebooks/*.
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
# Target spesifik file alert Wazuh terkompresi
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
}

# ── Feature Engineering Temporal (Data Preparation) ───────────────────
HOUR_GROUPS = {
    "dini_hari": range(0, 6),
    "pagi":      range(6, 12),
    "siang":     range(12, 18),
    "malam":     range(18, 24),
}

# ── Deduplikasi Temporal ──────────────────────────────────────────────
# Kolom kunci untuk mengelompokkan log berulang
DEDUP_GROUP_COLS    = ["agent_name", "rule_id", "decoder_name", "rule_level", "location"]
TIME_WINDOW_MINUTES = 10

# ── K-Modes Clustering (Agnostik Infrastruktur) ────────────────────────
CLUSTER_FEATURES    = ["rule_id", "rule_level", "decoder_name", "location", "hour_group"]
K_RANGE             = range(2, 11)
K_SEARCH_SAMPLE     = 20000   # jumlah baris sampel untuk pencarian K (Elbow & Silhouette)
KMODES_INIT         = "Huang"
KMODES_N_INIT       = 10
KMODES_RANDOM_STATE = 42
# Rule serangan web yang dipantau: di klaster mana rule ini jatuh?
WATCH_RULE_IDS      = ["31103", "31104", "31516", "31533", "100103", "100105"]

# ── Supervised Classification (Random Forest) ─────────────────────────
TARGET_LABEL_COL    = "label"
RF_TEST_SIZE        = 0.20
RF_RANDOM_STATE     = 42
RF_N_ESTIMATORS     = 200
RF_MAX_DEPTH        = 25
RF_CRITERION        = "gini"

# ── Output & Artifact Files ────────────────────────────────────────────
# Data hasil pemrosesan
PARQUET_CLEANED     = DATA_PROC_DIR / "cleaned.parquet"
PARQUET_DEDUPED     = DATA_PROC_DIR / "deduped.parquet"
PARQUET_CLUSTERED   = DATA_PROC_DIR / "clustered.parquet"
CSV_DEDUPED         = DATA_PROC_DIR / "deduped.csv"
CSV_CLUSTER_MAPPING = DATA_PROC_DIR / "cluster_mapping.csv"

# Laporan
CSV_PROFILES        = REPORTS_DIR / "cluster_profiles.csv"
CSV_K_SELECTION     = REPORTS_DIR / "k_selection.csv"
CONFIG_RUN_JSON     = REPORTS_DIR / "run_config.json"
CSV_CLUSTER_DETAIL  = REPORTS_DIR / "cluster_detail.csv"
CSV_CLUSTER_WATCH   = REPORTS_DIR / "cluster_watch_rules.csv"

# Model Artifacts (.pkl)
MODEL_KMODES_FILE   = MODELS_DIR / "kmodes_wazuh.pkl"
MODEL_RF_FILE       = MODELS_DIR / "random_forest_wazuh.pkl"
ENCODER_FILE        = MODELS_DIR / "custom_label_encoders.pkl"