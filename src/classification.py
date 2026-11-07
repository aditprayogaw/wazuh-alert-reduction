# src/classification.py
"""
Modul klasifikasi: Custom Label Encoder (UNKNOWN-safe), Random Forest, evaluasi,
serialisasi model, dan inferensi.

Perubahan dari versi sebelumnya:
- Fitur klasifikasi memakai CLASSIFICATION_FEATURES (tanpa rule_id, supaya model tidak
  sekadar menghafal tabel rule -> label).
- rule_level diperlakukan sebagai ANGKA ASLI (bukan teks yang diurutkan alfabetis).
- rule_id tetap dienkode, tetapi hanya untuk pencatatan "rule dikenal / rule baru".
- Inferensi: rule yang ada di katalog dilabeli lewat lookup; rule baru hanya diberi
  usulan model dan SELALU diserahkan ke analis (tidak pernah diredam otomatis).
- Evaluasi: tambahan macro-F1, balanced accuracy, MCC, dan kurva abstain.

Dipakai oleh: pipeline_b_train_model.py (Script B), pipeline_c_evaluate.py (Script C),
dan api/main.py (direncanakan, belum ada).
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split

from config import (
    CLASSIFICATION_FEATURES,
    NUMERIC_FEATURES,
    NEVER_SUPPRESS_LABELS,
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
CATEGORICAL_FEATURES = [c for c in CLASSIFICATION_FEATURES if c not in NUMERIC_FEATURES]


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


def fit_encoders(df_train: pd.DataFrame) -> dict:
    """Fit encoder untuk fitur kategorikal, rule_id (hanya pencatatan), dan target."""
    encoders = {}
    for col in CATEGORICAL_FEATURES + ["rule_id"]:
        encoders[col] = CustomLabelEncoder().fit(df_train[col])
    encoders[TARGET_LABEL_COL] = CustomLabelEncoder().fit(df_train[TARGET_LABEL_COL])
    return encoders


def _to_number(series: pd.Series) -> np.ndarray:
    """Fitur numerik dipakai apa adanya; nilai tidak terbaca menjadi -1."""
    return pd.to_numeric(series, errors="coerce").fillna(UNKNOWN_VALUE).to_numpy()


def encode_features(df: pd.DataFrame, encoders: dict) -> np.ndarray:
    """Encode fitur input. Kategorikal: nilai baru -> -1. Numerik: angka asli."""
    cols = []
    for col in CLASSIFICATION_FEATURES:
        if col in NUMERIC_FEATURES:
            cols.append(_to_number(df[col]))
        else:
            cols.append(encoders[col].transform(df[col]))
    return np.column_stack(cols)


def encode_target(df: pd.DataFrame, encoders: dict) -> np.ndarray:
    return encoders[TARGET_LABEL_COL].transform(df[TARGET_LABEL_COL])


def unknown_flags(df: pd.DataFrame, encoders: dict) -> dict:
    """Per fitur kategorikal: True bila nilainya belum pernah terlihat saat pelatihan."""
    return {c: encoders[c].transform(df[c]) == UNKNOWN_VALUE for c in CATEGORICAL_FEATURES}


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
    """Cetak laporan klasifikasi, confusion matrix, FPR per kelas, dan metrik tambahan."""
    target_enc = encoders[TARGET_LABEL_COL]
    class_names = target_enc.classes_
    labels = list(range(len(class_names)))

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    present = np.unique(np.concatenate([y_test, y_pred]))
    present = present[present >= 0]
    macro_f1 = f1_score(y_test, y_pred, labels=present, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f} | Balanced accuracy: {bal_acc:.4f} | MCC: {mcc:.4f}\n")
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
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
        "confusion_matrix": cm,
        "fpr": fpr,
    }


def abstain_curve(model, X, y, thresholds=(0.5, 0.7, 0.8, 0.9, 0.95)) -> pd.DataFrame:
    """
    Kompromi 'model memutuskan' vs 'diserahkan ke analis' pada berbagai ambang keyakinan.
    cakupan_% = porsi pola yang diputuskan model; sisanya diserahkan ke analis.
    """
    proba = model.predict_proba(X)
    conf = proba.max(axis=1)
    pred = model.classes_[proba.argmax(axis=1)]
    rows = []
    for t in thresholds:
        m = conf >= t
        rows.append({
            "ambang": t,
            "cakupan_%": round(float(m.mean() * 100), 2),
            "akurasi_pada_yang_diputuskan": round(float((pred[m] == y[m]).mean()), 4) if m.any() else None,
            "diserahkan_ke_analis_%": round(float((~m).mean() * 100), 2),
        })
    return pd.DataFrame(rows)


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
    plt.close(fig)  # backend Agg: tidak perlu plt.show()


# ── Serialisasi ────────────────────────────────────────────────────────
def save_artifacts(model, encoders: dict) -> None:
    joblib.dump(model, MODEL_RF_FILE)
    joblib.dump(encoders, ENCODER_FILE)


def load_artifacts():
    return joblib.load(MODEL_RF_FILE), joblib.load(ENCODER_FILE)


# ── Inferensi (direncanakan dipakai FastAPI) ───────────────────────────
def predict_records(records: list, model, encoders: dict, rule_labels: dict = None) -> list:
    """
    Prediksi untuk list dict berisi 'rule_id' + CLASSIFICATION_FEATURES.

    Kebijakan konservatif:
    - rule ada di katalog  -> label dari katalog (lookup, bukan tebakan model).
      Boleh diredam, kecuali labelnya termasuk NEVER_SUPPRESS_LABELS.
    - rule belum ada di katalog -> hanya usulan model; TIDAK PERNAH diredam otomatis,
      selalu perlu_review.
    """
    df = pd.DataFrame(records).astype(str)
    X = encode_features(df, encoders)
    proba = model.predict_proba(X)
    pred = model.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    names = encoders[TARGET_LABEL_COL].inverse_transform(pred)

    flags = unknown_flags(df, encoders)
    unk = np.column_stack(list(flags.values())).any(axis=1) if flags else np.zeros(len(df), dtype=bool)
    rule_labels = rule_labels or {}

    out = []
    for i, rid in enumerate(df["rule_id"]):
        if rid in rule_labels:
            label, sumber, keyakinan = rule_labels[rid], "katalog", 1.0
            boleh = label not in NEVER_SUPPRESS_LABELS
        else:
            label, sumber, keyakinan = names[i], "model", float(conf[i])
            boleh = False
        out.append({
            "label": label,
            "sumber": sumber,
            "keyakinan": round(keyakinan, 3),
            "nilai_asing": bool(unk[i]),
            "boleh_diredam": boleh,
            "perlu_review": not boleh,
        })
    return out