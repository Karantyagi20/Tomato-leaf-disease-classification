"""
predict.py
──────────
Inference script for both the Random Forest and K-Means models.
Accepts a single leaf image path and returns the predicted disease category.

Usage
-----
    python predict.py --image path/to/leaf.jpg --model rf
    python predict.py --image path/to/leaf.jpg --model kmeans

Author : Karan
Course : B.Tech — Artificial Intelligence & Machine Learning
College : Manipal University Jaipur
Year   : 2025-26
"""

import argparse
import cv2
import numpy as np


IMAGE_SIZE = (64, 64)


def preprocess_single_image(image_path: str, image_size: tuple = IMAGE_SIZE) -> np.ndarray:
    """Load, resize, convert, and flatten a single image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img = cv2.resize(img, image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.flatten().reshape(1, -1)


def predict_rf(image_path: str, model, pca, le) -> str:
    """
    Predict disease class using the trained Random Forest model.

    Parameters
    ----------
    image_path : str               – path to the input leaf image
    model      : RandomForestClassifier
    pca        : PCA               – fitted PCA transformer
    le         : LabelEncoder      – fitted label encoder

    Returns
    -------
    str – predicted disease category name
    """
    img_flat = preprocess_single_image(image_path)
    img_pca  = pca.transform(img_flat)
    pred     = model.predict(img_pca)[0]
    proba    = model.predict_proba(img_pca)[0]
    label    = le.inverse_transform([pred])[0]
    conf     = proba.max() * 100
    print(f"[RF]     Predicted : {label}  ({conf:.1f}% confidence)")
    return label


def predict_kmeans(image_path: str, kmeans, pca, le, cluster_to_label: dict) -> str:
    """
    Predict disease class using the K-Means clustering model.

    Parameters
    ----------
    image_path       : str        – path to the input leaf image
    kmeans           : KMeans     – fitted K-Means model
    pca              : PCA        – fitted PCA transformer
    le               : LabelEncoder
    cluster_to_label : dict       – cluster_id → encoded label

    Returns
    -------
    str – predicted disease category name
    """
    img_flat = preprocess_single_image(image_path)
    img_pca  = pca.transform(img_flat)
    cluster  = kmeans.predict(img_pca)[0]
    label_id = cluster_to_label[cluster]
    label    = le.inverse_transform([label_id])[0]
    print(f"[KMeans] Predicted : {label}  (Cluster {cluster})")
    return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tomato Disease Prediction")
    parser.add_argument("--image", required=True, help="Path to leaf image")
    parser.add_argument("--model", choices=["rf", "kmeans"], default="rf",
                        help="Which model to use: 'rf' or 'kmeans'")
    args = parser.parse_args()

    print(f"\nPredicting with model: {args.model.upper()}")
    print(f"Image: {args.image}\n")

    if args.model == "rf":
        from train_model import train
        best_model, le, pca = train()
        predict_rf(args.image, best_model, pca, le)
    else:
        from kmeans_model import train_kmeans
        kmeans, pca, le, cluster_to_label = train_kmeans()
        predict_kmeans(args.image, kmeans, pca, le, cluster_to_label)
