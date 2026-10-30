# src/classification.py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from config import (
    CLUSTER_FEATURES,
    TARGET_LABEL_COL,
    RF_TEST_SIZE,
    RF_RANDOM_STATE,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_CRITERION,
    MODEL_RF_FILE,
    ENCODER_FILE,
    MODELS_DIR,
    FIGURES_DIR,
    CSV_CLUSTER_MAPPING,
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

UNKNOWN_VALUE = -1


# ── Custom Label Encoder (UNKNOWN-safe) ────────────────────────────────
class CustomLabelEncoder:
    """Label encoder dengan penanganan nilai tak dikenal (-1) saat inferensi."""

    def __init__(self):
        self.classes_ = []
        self._map = {}

    def fit(self, values):
        self.classes_ = sorted(pd.Series(values).astype(str).unique().tolist())
        self._map = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, values) -> np.ndarray:
        return np.array(
            [self._map.get(str(v), UNKNOWN_VALUE) for v in values], dtype=int
        )

    def inverse_transform(self, codes) -> list:
        return [
            self.classes_[c] if 0 <= c < len(self.classes_) else "UNKNOWN"
            for c in codes
        ]


# ── Pelabelan dari Mapping Manual ──────────────────────────────────────
def apply_cluster_labels(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """Sebarkan label_semantik dari cluster_mapping.csv ke seluruh baris data."""
    mapping = pd.read_csv(CSV_CLUSTER_MAPPING)

    if "label_semantik" not in mapping.columns:
        raise ValueError("Kolom 'label_semantik' tidak ada di cluster_mapping.csv")

    mapping["label_semantik"] = mapping["label_semantik"].astype(str).str.strip()
    empty = mapping[
        mapping["label_semantik"].isin(["", "nan"])
    ]["cluster"].tolist()
    if empty:
        raise ValueError(
            f"Label semantik belum diisi untuk cluster: {empty}. "
            f"Lengkapi {CSV_CLUSTER_MAPPING} terlebih dahulu."
        )

    label_map = dict(zip(mapping["cluster"].astype(int), mapping["label_semantik"]))
    df = df_clustered.copy()
    df[TARGET_LABEL_COL] = df["cluster"].astype(int).map(label_map)

    if df[TARGET_LABEL_COL].isna().any():
        missing = df.loc[df[TARGET_LABEL_COL].isna(), "cluster"].unique().tolist()
        raise ValueError(f"Cluster {missing} tidak ditemukan di mapping.")
    return df


# ── Split & Encoding ───────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    """Stratified split 80/20 berdasarkan label."""
    counts = df[TARGET_LABEL_COL].value_counts()
    stratify = df[TARGET_LABEL_COL] if counts.min() >= 2 else None
    if stratify is None:
        print("PERINGATAN: ada kelas dengan <2 sampel, stratified split dinonaktifkan.")
    return train_test_split(
        df,
        test_size=RF_TEST_SIZE,
        random_state=RF_RANDOM_STATE,
        stratify=stratify,
    )


def fit_encoders(df_train: pd.DataFrame) -> dict:
    """Fit encoder untuk setiap fitur input dan target."""
    encoders = {}
    for col in CLUSTER_FEATURES:
        encoders[col] = CustomLabelEncoder().fit(df_train[col])
    encoders[TARGET_LABEL_COL] = CustomLabelEncoder().fit(df_train[TARGET_LABEL_COL])
    return encoders


def encode_features(df: pd.DataFrame, encoders: dict) -> np.ndarray:
    """Encode fitur input; nilai baru menjadi -1."""
    return np.column_stack([
        encoders[col].transform(df[col]) for col in CLUSTER_FEATURES
    ])


def encode_target(df: pd.DataFrame, encoders: dict) -> np.ndarray:
    return encoders[TARGET_LABEL_COL].transform(df[TARGET_LABEL_COL])


# ── Training ───────────────────────────────────────────────────────────
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        criterion=RF_CRITERION,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ── Evaluasi ───────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, encoders: dict) -> dict:
    """Cetak laporan klasifikasi, confusion matrix, dan FPR per kelas."""
    target_enc = encoders[TARGET_LABEL_COL]
    class_names = target_enc.classes_
    labels = list(range(len(class_names)))

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred, labels=labels, target_names=class_names, zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))

    # False Positive Rate per kelas: FP / (FP + TN)
    fpr = {}
    total = cm.sum()
    print("\nFalse Positive Rate per kelas:")
    for i, name in enumerate(class_names):
        fp = cm[:, i].sum() - cm[i, i]
        tn = total - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fpr[name] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        print(f"  {name}: {fpr[name]:.4f}")

    plot_confusion_matrix(cm, class_names)
    return {"accuracy": acc, "confusion_matrix": cm, "fpr": fpr}


def plot_confusion_matrix(cm: np.ndarray, class_names: list) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix — Random Forest")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_rf.png", dpi=150)
    plt.show()


# ── Serialisasi ────────────────────────────────────────────────────────
def save_artifacts(model, encoders: dict) -> None:
    joblib.dump(model, MODEL_RF_FILE)
    joblib.dump(encoders, ENCODER_FILE)


def load_artifacts():
    return joblib.load(MODEL_RF_FILE), joblib.load(ENCODER_FILE)


# ── Inferensi (dipakai FastAPI) ────────────────────────────────────────
def predict_records(records: list, model, encoders: dict) -> list:
    """Prediksi label untuk list dict berisi fitur CLUSTER_FEATURES."""
    df = pd.DataFrame(records)[CLUSTER_FEATURES].astype(str)
    X = encode_features(df, encoders)
    y_pred = model.predict(X)
    return encoders[TARGET_LABEL_COL].inverse_transform(y_pred)