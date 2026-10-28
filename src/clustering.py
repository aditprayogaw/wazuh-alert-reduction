# src/clustering.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import prince
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

from config import (
    CLUSTER_FEATURES,
    K_RANGE,
    KMODES_INIT,
    KMODES_N_INIT,
    KMODES_RANDOM_STATE,
    FIGURES_DIR,
)


# ── Elbow Method ───────────────────────────────────────────────────────
def compute_elbow(X: np.ndarray) -> list:
    """Hitung cost K-Modes untuk setiap K dalam K_RANGE."""
    costs = []
    for k in K_RANGE:
        km = KModes(
            n_clusters=k,
            init=KMODES_INIT,
            n_init=KMODES_N_INIT,
            random_state=KMODES_RANDOM_STATE,
        )
        km.fit(X)
        costs.append(km.cost_)
        print(f"K={k:2d} | cost={km.cost_:.2f}")
    return costs


def plot_elbow(costs: list) -> None:
    """Visualisasi Elbow Method."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(list(K_RANGE), costs, marker="o", linewidth=2)
    ax.set_title("Elbow Method — K-Modes Cost per K")
    ax.set_xlabel("Jumlah Cluster (K)")
    ax.set_ylabel("Cost (Dissimilarity)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "elbow_kmodes.png", dpi=150)
    plt.show()


# ── Silhouette Score ───────────────────────────────────────────────────
def compute_silhouette(X: np.ndarray) -> list:
    """Hitung Silhouette Score untuk setiap K (metric Hamming)."""
    X_encoded = np.column_stack([
        LabelEncoder().fit_transform(X[:, i])
        for i in range(X.shape[1])
    ])

    scores = []
    for k in K_RANGE:
        km = KModes(
            n_clusters=k,
            init=KMODES_INIT,
            n_init=KMODES_N_INIT,
            random_state=KMODES_RANDOM_STATE,
        )
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            scores.append(None)
            continue
        score = silhouette_score(X_encoded, labels, metric="hamming")
        scores.append(score)
        print(f"K={k:2d} | silhouette={score:.4f}")
    return scores


def plot_silhouette(scores: list) -> None:
    """Visualisasi Silhouette Score."""
    valid_k = [k for k, s in zip(K_RANGE, scores) if s is not None]
    valid_s = [s for s in scores if s is not None]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(valid_k, valid_s, marker="o", linewidth=2, color="darkorange")
    ax.set_title("Silhouette Score per K — K-Modes (Hamming)")
    ax.set_xlabel("Jumlah Cluster (K)")
    ax.set_ylabel("Silhouette Score")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "silhouette_kmodes.png", dpi=150)
    plt.show()


# ── Model Final ────────────────────────────────────────────────────────

def fit_kmodes(X: np.ndarray, k: int) -> KModes:
    """Fit K-Modes dengan K optimal."""
    km = KModes(
        n_clusters=k,
        init=KMODES_INIT,
        n_init=KMODES_N_INIT,
        random_state=KMODES_RANDOM_STATE,
    )
    km.fit(X)
    return km


# ── MCA Visualization ──────────────────────────────────────────────────

def plot_mca(df: pd.DataFrame, cluster_labels: dict,
             cluster_colors: dict, title: str, filename: str) -> None:
    """Visualisasi MCA Cluster Map."""
    mca = prince.MCA(n_components=2, random_state=42)
    mca = mca.fit(df[CLUSTER_FEATURES].astype(str))
    coords = mca.transform(df[CLUSTER_FEATURES].astype(str))
    coords.columns = ["Dim1", "Dim2"]
    coords["cluster"] = df["cluster"].values

    fig, ax = plt.subplots(figsize=(10, 7))
    for cid in sorted(coords["cluster"].unique()):
        subset = coords[coords["cluster"] == cid]
        ax.scatter(
            subset["Dim1"], subset["Dim2"],
            label=f"Cluster {cid} — {cluster_labels.get(cid, '')}",
            color=cluster_colors.get(cid, "gray"),
            alpha=0.7, s=60,
        )
    ax.set_title(title)
    ax.set_xlabel("Dimensi 1")
    ax.set_ylabel("Dimensi 2")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.show()

    # Variance explained
    explained = mca.percentage_of_variance_
    print("Variance explained per dimensi MCA:")
    for i, var in enumerate(explained):
        print(f"  Dimensi {i+1} : {var:.2f}%")
    print(f"  Total     : {sum(explained):.2f}%")