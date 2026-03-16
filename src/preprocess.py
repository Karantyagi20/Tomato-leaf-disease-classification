"""
preprocess.py
─────────────
Image loading and preprocessing pipeline for the Tomato Disease Detection project.
Handles resizing, colour conversion, flattening, and label encoding.

Author : Karan
Course : B.Tech — Artificial Intelligence & Machine Learning
College : Manipal University Jaipur
Year   : 2025-26
"""

import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ── Constants ──────────────────────────────────────────────────────────────
IMAGE_SIZE   = (64, 64)
DATASET_PATH = "dataset/"          # root folder containing class sub-directories


def load_dataset(path: str = DATASET_PATH, image_size: tuple = IMAGE_SIZE):
    """
    Load all images from a folder-per-class dataset structure.

    Expected directory layout:
        path/
        ├── Tomato_Early_blight/
        ├── Tomato_Late_blight/
        ├── Tomato_Leaf_Mold/
        └── Tomato_healthy/

    Parameters
    ----------
    path       : str   – root directory of the dataset
    image_size : tuple – (width, height) to resize every image

    Returns
    -------
    data           : np.ndarray of shape (N, W*H*3)  – flattened pixel vectors
    labels         : np.ndarray of shape (N,)         – string class labels
    labels_encoded : np.ndarray of shape (N,)         – integer-encoded labels
    le             : LabelEncoder                      – fitted encoder
    label_names    : list[str]                         – sorted class names
    """
    data         = []
    labels       = []
    label_names  = sorted(os.listdir(path))

    for label in label_names:
        folder_path = os.path.join(path, label)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue                            # skip unreadable files
            img = cv2.resize(img, image_size)       # resize to fixed size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB
            data.append(img.flatten())              # flatten to 1D vector
            labels.append(label)

    data   = np.array(data)
    labels = np.array(labels)

    le             = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    print(f"[preprocess] Loaded {len(data)} images from {len(label_names)} classes")
    print(f"[preprocess] Classes: {label_names}")
    print(f"[preprocess] Feature vector size: {data.shape[1]}D")

    return data, labels, labels_encoded, le, label_names


def apply_pca(data: np.ndarray, n_components: int = 120, random_state: int = 42):
    """
    Apply PCA dimensionality reduction to flattened image data.

    Parameters
    ----------
    data         : np.ndarray – raw pixel vectors (N, D)
    n_components : int        – number of PCA components to retain
    random_state : int        – reproducibility seed

    Returns
    -------
    data_pca : np.ndarray – reduced data (N, n_components)
    pca      : PCA        – fitted PCA object
    """
    from sklearn.decomposition import PCA

    print(f"[preprocess] Applying PCA: {data.shape[1]}D → {n_components}D ...")
    pca      = PCA(n_components=n_components, random_state=random_state)
    data_pca = pca.fit_transform(data)

    variance = pca.explained_variance_ratio_.sum() * 100
    print(f"[preprocess] Variance retained: {variance:.2f}% with {n_components} components")

    return data_pca, pca
