# src/classification.py
"""
Modul Pemodelan Klasifikasi Triase SOC (Random Forest).

Fitur:
1. Custom Label Encoder: Menangani kategori asing/baru saat inferensi (UNKNOWN-safe -> -1).
2. Pemetaan Fitur Konteks: rule_level sebagai numerik asli, decoder_name, location, rule_groups.
3. Model Training & Serialization: Random Forest dengan class_weight balanced.
4. Logika Triase Rule Baru:
   - Dismiss hanya jika probabilitas model >= CONFIDENCE_THRESHOLD_DISMISS.
   - Selain itu dialihkan ke Attention (Review + Escalate) demi keamanan SOC.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import (
    CLASSIFICATION_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_LABEL_COL,
    RF_RANDOM_STATE,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_CRITERION,
    RF_CLASS_WEIGHT,
    MODEL_RF_FILE,
    ENCODER_FILE,
    MODELS_DIR,
    CONFIDENCE_THRESHOLD_DISMISS,
)
from src.labeling import load_rule_labels

MODELS_DIR.mkdir(parents=True, exist_ok=True)
UNKNOWN_VALUE = -1


class CustomLabelEncoder:
    """Label encoder yang aman terhadap nilai tak dikenal (-1) saat inferensi rule baru."""
    def __init__(self):
        self.classes_ = []
        self._map = {}

    def fit(self, values):
        self.classes_ = sorted(pd.Series(values).astype(str).unique().tolist())
        self._map = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, values) -> np.ndarray:
        return np.array([self._map.get(str(v), UNKNOWN_VALUE) for v in values], dtype=int)

    def inverse_transform(self, codes) -> list:
        return [self.classes_[c] if 0 <= c < len(self.classes_) else "UNKNOWN" for c in codes]


def apply_rule_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghubungkan tabel log terdeduplikasi dengan tabel kebijakan rule_labels_done.csv
    berdasarkan kolom rule_id.
    """
    label_map = load_rule_labels()
    out = df.copy()
    out[TARGET_LABEL_COL] = out["rule_id"].astype(str).map(label_map)

    if out[TARGET_LABEL_COL].isna().any():
        missing = sorted(out.loc[out[TARGET_LABEL_COL].isna(), "rule_id"].astype(str).unique())
        raise ValueError(f"Ditemukan rule_id tanpa label di tabel kebijakan: {missing}.")
    return out


def fit_encoders(df_train: pd.DataFrame) -> dict:
    """Melatih CustomLabelEncoder untuk setiap fitur kategorikal dan label target."""
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        encoders[col] = CustomLabelEncoder().fit(df_train[col])
    encoders[TARGET_LABEL_COL] = CustomLabelEncoder().fit(df_train[TARGET_LABEL_COL])
    return encoders


def encode_features(df: pd.DataFrame, encoders: dict) -> np.ndarray:
    """Transformasi fitur konteks X (numerik + kategorikal)."""
    columns_data = []
    for col in CLASSIFICATION_FEATURES:
        if col in NUMERIC_FEATURES:
            columns_data.append(pd.to_numeric(df[col], errors="coerce").fillna(0).values)
        else:
            columns_data.append(encoders[col].transform(df[col]))
    return np.column_stack(columns_data)


def encode_target(df: pd.DataFrame, encoders: dict) -> np.ndarray:
    """Transformasi target Y."""
    return encoders[TARGET_LABEL_COL].transform(df[TARGET_LABEL_COL])


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Melatih Random Forest Classifier."""
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        criterion=RF_CRITERION,
        class_weight=RF_CLASS_WEIGHT,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict_new_rules_triase(model: RandomForestClassifier, X: np.ndarray, encoders: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Prediksi triase khusus untuk RULE BARU (Unknown Rules):
    Mengembalikan: (predicted_labels, max_probabilities).
    
    Aturan Keamanan SOC:
    - Jika prediksi 'Dismiss' tetapi confidence < CONFIDENCE_THRESHOLD_DISMISS -> diubah menjadi 'Attention'.
    - Prediksi 'Review' atau 'Escalate' -> tetap sebagai 'Attention' (membutuhkan perhatian manusia).
    """
    probs = model.predict_proba(X)
    target_encoder = encoders[TARGET_LABEL_COL]
    class_indices = {cls_name: idx for idx, cls_name in enumerate(target_encoder.classes_)}
    dismiss_idx = class_indices.get("Dismiss")

    final_predictions = []
    max_confidences = np.max(probs, axis=1)

    for i in range(len(X)):
        # Jika model menebak Dismiss
        if dismiss_idx is not None and np.argmax(probs[i]) == dismiss_idx:
            if probs[i][dismiss_idx] >= CONFIDENCE_THRESHOLD_DISMISS:
                final_predictions.append("Dismiss")
            else:
                final_predictions.append("Attention")  # Ambigu, alihkan ke analis
        else:
            # Jika tebakan Review atau Escalate
            final_predictions.append("Attention")

    return np.array(final_predictions), max_confidences


def save_artifacts(model, encoders: dict) -> None:
    """Menyimpan artefak model dan encoder."""
    joblib.dump(model, MODEL_RF_FILE)
    joblib.dump(encoders, ENCODER_FILE)


def load_artifacts():
    """Memuat artefak model dan encoder."""
    return joblib.load(MODEL_RF_FILE), joblib.load(ENCODER_FILE)
