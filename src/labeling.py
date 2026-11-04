# src/labeling.py
"""
Pelabelan semantik per rule_id (Opsi 4: klaster mengusulkan, pakar mengoreksi).

- label_saran : usulan otomatis dari grup bawaan Wazuh (aturan di config.LABEL_GROUP_RULES)
- label_final : label yang dipakai pipeline. Awalnya = label_saran, lalu dikoreksi pakar
                langsung di data/processed/rule_labels.csv
"""
import pandas as pd

from config import (
    CSV_RULE_LABELS,
    LABEL_GROUP_RULES,
    LABEL_RULE_ID_OVERRIDES,
    LABEL_DEFAULT,
)


def _tokens(groups) -> set:
    return {t.strip() for t in str(groups).split(",") if t.strip()}


def suggest_label(rule_id, groups) -> str:
    """Usulan label dari grup Wazuh (urutan prioritas mengikuti LABEL_GROUP_RULES)."""
    rid = str(rule_id)
    if rid in LABEL_RULE_ID_OVERRIDES:
        return LABEL_RULE_ID_OVERRIDES[rid]
    tokens = _tokens(groups)
    for label, keys in LABEL_GROUP_RULES:
        if tokens & keys:
            return label
    return LABEL_DEFAULT


def build_rule_table(df: pd.DataFrame) -> pd.DataFrame:
    """Katalog rule (deskripsi, level, grup, total alert, sebaran klaster) + label saran."""
    base = df.groupby("rule_id").agg(
        deskripsi=("rule_description", "first"),
        level=("rule_level", "first"),
        groups=("rule_groups", "first"),
        alert=("alert_count", "sum"),
    )
    if "cluster" in df.columns:
        dist = df.pivot_table(index="rule_id", columns="cluster",
                              values="alert_count", aggfunc="sum", fill_value=0)
        dist.columns = [f"cluster_{c}" for c in dist.columns]
        base = base.join(dist)
    table = base.reset_index()
    table["rule_id"] = table["rule_id"].astype(str)
    table["label_saran"] = [suggest_label(r, g) for r, g in zip(table["rule_id"], table["groups"])]
    return table.sort_values("alert", ascending=False).reset_index(drop=True)


def export_rule_labels(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """Tulis rule_labels.csv. Koreksi pakar pada file lama (label_final) tetap dipertahankan."""
    table = build_rule_table(df_clustered)
    table["label_final"] = table["label_saran"]

    if CSV_RULE_LABELS.exists():
        old = pd.read_csv(CSV_RULE_LABELS, dtype={"rule_id": str})
        kept = {
            r: str(l).strip()
            for r, l in zip(old["rule_id"], old["label_final"])
            if pd.notna(l) and str(l).strip()
        }
        table["label_final"] = [kept.get(r, s) for r, s in zip(table["rule_id"], table["label_saran"])]

    CSV_RULE_LABELS.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV_RULE_LABELS, index=False)
    return table


def load_rule_labels() -> dict:
    """Baca rule_labels.csv -> {rule_id: label_final}."""
    if not CSV_RULE_LABELS.exists():
        raise FileNotFoundError(f"{CSV_RULE_LABELS} belum ada. Jalankan Script A terlebih dahulu.")
    table = pd.read_csv(CSV_RULE_LABELS, dtype={"rule_id": str})
    table["label_final"] = table["label_final"].fillna("").astype(str).str.strip()
    empty = table.loc[table["label_final"] == "", "rule_id"].tolist()
    if empty:
        raise ValueError(f"label_final kosong untuk rule_id: {empty}. Lengkapi {CSV_RULE_LABELS}.")
    return dict(zip(table["rule_id"], table["label_final"]))