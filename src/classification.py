# src/classification.py
"""
Modul klasifikasi: pelabelan per rule_id, Custom Label Encoder (UNKNOWN-safe),
Random Forest, evaluasi, serialisasi model, dan inferensi.

Dipakai oleh: pipeline_b_train_model.py (Script B), pipeline_c_evaluate.py (Script C),
dan api/main.py (inferensi).
"""
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
    CLASSIFICATION_FEATURES,
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
)
from src.labeling import load_rule_labels

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


# ── Pelabelan dari rule_labels.csv ─────────────────────────────────────
def apply_rule_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Beri label ke setiap baris berdasarkan rule_id (sumber: rule_labels.csv, kolom label_final)."""
    label_map = load_rule_labels()
    out = df.copy()
    out[TARGET_LABEL_COL] = out["rule_id"].astype(str).map(label_map)
    if out[TARGET_LABEL_COL].isna().any():
        missing = sorted(out.loc[out[TARGET_LABEL_COL].isna(), "rule_id"].astype(str).unique())
        raise ValueError(f"rule_id belum ada di rule_labels.csv: {missing}. Jalankan ulang Script A.")
    return out


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


def fit_encoders(df_train):
    encoders = {}
    # rule_id hanya untuk pencatatan "rule dikenal", BUKAN fitur model
    for col in CLASSIFICATION_FEATURES + ["rule_id"]:
        encoders[col] = CustomLabelEncoder().fit(df_train[col])
    encoders[TARGET_LABEL_COL] = CustomLabelEncoder().fit(df_train[TARGET_LABEL_COL])
    return encoders


def encode_features(df, encoders):
    return np.column_stack([encoders[c].transform(df[c]) for c in CLASSIFICATION_FEATURES])


def encode_target(df: pd.DataFrame, encoders: dict) -> np.ndarray:
    return encoders[TARGET_LABEL_COL].transform(df[TARGET_LABEL_COL])


# ── Training ───────────────────────────────────────────────────────────
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        criterion=RF_CRITERION,
        class_weight="balanced",
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ── Evaluasi ───────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, encoders: dict,
                   fig_name: str = "confusion_matrix_rf.png") -> dict:
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

    plot_confusion_matrix(cm, class_names, fig_name)
    return {"accuracy": acc, "confusion_matrix": cm, "fpr": fpr}


def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                          fig_name: str = "confusion_matrix_rf.png") -> None:
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
    plt.savefig(FIGURES_DIR / fig_name, dpi=150)
    plt.show()


# ── Serialisasi ────────────────────────────────────────────────────────
def save_artifacts(model, encoders: dict) -> None:
    joblib.dump(model, MODEL_RF_FILE)
    joblib.dump(encoders, ENCODER_FILE)


def load_artifacts():
    return joblib.load(MODEL_RF_FILE), joblib.load(ENCODER_FILE)


# ── Inferensi (dipakai FastAPI) ────────────────────────────────────────
def predict_records(records, model, encoders, rule_labels=None):
    df = pd.DataFrame(records).astype(str)
    X = encode_features(df, encoders)
    ml = encoders[TARGET_LABEL_COL].inverse_transform(model.predict(X))
    unk = (X == -1).any(axis=1)
    out = []
    for rid, pred, u in zip(df["rule_id"], ml, unk):
        if rule_labels and rid in rule_labels:
            out.append({"label": rule_labels[rid], "sumber": "katalog", "perlu_review": False})
        else:
            out.append({"label": pred, "sumber": "model", "perlu_review": True})
    return out