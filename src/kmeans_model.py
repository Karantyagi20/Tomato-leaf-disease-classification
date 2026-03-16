"""
kmeans_model.py
───────────────
K-Means Clustering pipeline with PCA dimensionality reduction and
majority-vote cluster-to-label mapping for unsupervised disease classification.

Author : Karan
Course : B.Tech — Artificial Intelligence & Machine Learning
College : Manipal University Jaipur
Year   : 2025-26
"""

import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocess import load_dataset, apply_pca


# ── Config ─────────────────────────────────────────────────────────────────
PCA_COMPS    = 120
N_INIT       = 10
MAX_ITER     = 300
RANDOM_STATE = 42


def train_kmeans(dataset_path: str = "dataset/"):
    """
    Full K-Means clustering pipeline:
      1. Load & preprocess all images (no train/test split — unsupervised)
      2. Apply PCA (120 components, ~81.37% variance)
      3. Fit K-Means (k = number of disease classes)
      4. Map clusters → disease labels via majority vote
      5. Evaluate accuracy, print classification report

    Parameters
    ----------
    dataset_path : str – path to the image dataset root directory

    Returns
    -------
    kmeans          : KMeans         – fitted K-Means model
    pca             : PCA            – fitted PCA transformer
    le              : LabelEncoder   – fitted label encoder
    cluster_to_label: dict           – cluster_id → encoded label mapping
    """
    # ── 1. Load data ──────────────────────────────────────────────────────
    data, labels, labels_encoded, le, label_names = load_dataset(dataset_path)
    n_clusters = len(label_names)

    # ── 2. PCA ────────────────────────────────────────────────────────────
    data_pca, pca = apply_pca(data, n_components=PCA_COMPS)

    # ── 3. K-Means ────────────────────────────────────────────────────────
    print(f"[kmeans] Running K-Means with k={n_clusters} clusters ...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",       # smarter centroid initialisation
        n_init=N_INIT,          # run N_INIT times, pick best inertia
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    cluster_labels = kmeans.fit_predict(data_pca)
    print("[kmeans] Clustering complete.")

    # ── 4. Majority-vote cluster → label mapping ──────────────────────────
    cluster_to_label = {}
    for cluster_id in range(n_clusters):
        mask                        = cluster_labels == cluster_id
        true_labels_in_cluster      = labels_encoded[mask]
        most_common                 = mode(true_labels_in_cluster, keepdims=True).mode[0]
        cluster_to_label[cluster_id] = most_common

    print("\n[kmeans] Cluster → Disease mapping:")
    for cid, lid in cluster_to_label.items():
        print(f"         Cluster {cid} → {le.inverse_transform([lid])[0]}")

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    y_pred = np.array([cluster_to_label[c] for c in cluster_labels])
    acc    = accuracy_score(labels_encoded, y_pred)

    print(f"\n[kmeans] ── Results ─────────────────────────────")
    print(f"[kmeans] Clustering Accuracy : {acc * 100:.2f}%")
    print(f"\n[kmeans] Classification Report:\n")
    print(classification_report(labels_encoded, y_pred, target_names=le.classes_))

    return kmeans, pca, le, cluster_to_label


if __name__ == "__main__":
    train_kmeans()
