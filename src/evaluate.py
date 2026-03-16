"""
evaluate.py
───────────
Evaluation utilities — confusion matrix, classification report, and
comparison charts for both Random Forest and K-Means models.

Author : Karan
Course : B.Tech — Artificial Intelligence & Machine Learning
College : Manipal University Jaipur
Year   : 2025-26
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from matplotlib.colors import LinearSegmentedColormap


NAVY  = '#0D2137'
BLUE1 = '#1565C0'
BLUE2 = '#1E88E5'
BLUE3 = '#42A5F5'
BLUE4 = '#BBDEFB'
WHITE = '#FFFFFF'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': 'white'})


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix",
                          cmap_colors=None, save_path=None):
    """
    Plot and optionally save a styled confusion matrix heatmap.

    Parameters
    ----------
    y_true      : array-like – true labels (encoded)
    y_pred      : array-like – predicted labels (encoded)
    class_names : list[str]  – display names for classes
    title       : str        – plot title
    cmap_colors : list       – custom colour gradient (default: blue)
    save_path   : str|None   – if provided, saves figure to this path
    """
    cm     = confusion_matrix(y_true, y_pred)
    colors = cmap_colors or [WHITE, BLUE4, BLUE2, BLUE1, NAVY]
    cmap   = LinearSegmentedColormap.from_list('custom', colors)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold', color=NAVY)
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold', color=NAVY)
    ax.set_title(title, fontsize=13, fontweight='bold', color=NAVY, pad=12)

    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > thresh else NAVY)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[evaluate] Saved confusion matrix → {save_path}")
    plt.show()
    plt.close()


def print_report(y_true, y_pred, class_names):
    """Print accuracy and full classification report."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy : {acc * 100:.2f}%\n")
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_cluster_scatter(data_2d, cluster_labels, title="K-Means Clusters (PCA 2D)",
                         save_path=None):
    """
    Visualise K-Means cluster assignments in 2D PCA space.

    Parameters
    ----------
    data_2d        : np.ndarray (N, 2) – 2D PCA projections
    cluster_labels : np.ndarray (N,)   – cluster IDs per sample
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(data_2d[:, 0], data_2d[:, 1],
                    c=cluster_labels, cmap='tab10', alpha=0.5, s=12)
    plt.colorbar(sc, ax=ax, label='Cluster ID')
    ax.set_xlabel('PCA Component 1', fontsize=12, fontweight='bold', color=NAVY)
    ax.set_ylabel('PCA Component 2', fontsize=12, fontweight='bold', color=NAVY)
    ax.set_title(title, fontsize=14, fontweight='bold', color=NAVY, pad=12)
    for sp in ax.spines.values():
        sp.set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
