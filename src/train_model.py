"""
train_model.py
──────────────
Training pipeline for the Random Forest Classifier with GridSearchCV
hyperparameter tuning for the Tomato Disease Detection project.

Author : Karan
Course : B.Tech — Artificial Intelligence & Machine Learning
College : Manipal University Jaipur
Year   : 2025-26
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocess import load_dataset, apply_pca


# ── Hyperparameter grid ────────────────────────────────────────────────────
PARAM_GRID = {
    "n_estimators":     [50, 100, 200, 300],
    "max_depth":        [None, 10, 20],
    "min_samples_leaf": [1, 2],
    "min_samples_split":[2, 5],
}

TEST_SIZE    = 0.20
RANDOM_STATE = 42
CV_FOLDS     = 5
PCA_COMPS    = 120


def train(dataset_path: str = "dataset/"):
    """
    Full Random Forest training pipeline:
      1. Load & preprocess images
      2. Apply PCA
      3. Train/test split (stratified)
      4. GridSearchCV hyperparameter tuning
      5. Evaluate on test set

    Parameters
    ----------
    dataset_path : str – path to the image dataset root directory

    Returns
    -------
    best_model : RandomForestClassifier – trained & tuned model
    le         : LabelEncoder           – fitted label encoder
    pca        : PCA                    – fitted PCA transformer
    """
    # ── 1. Load data ──────────────────────────────────────────────────────
    data, labels, labels_encoded, le, label_names = load_dataset(dataset_path)

    # ── 2. PCA ────────────────────────────────────────────────────────────
    data_pca, pca = apply_pca(data, n_components=PCA_COMPS)

    # ── 3. Train / test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        data_pca, labels_encoded,
        test_size=TEST_SIZE,
        stratify=labels_encoded,
        random_state=RANDOM_STATE,
    )
    print(f"[train] Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # ── 4. GridSearchCV ───────────────────────────────────────────────────
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=PARAM_GRID,
        cv=CV_FOLDS,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1,
    )

    print(f"[train] Running GridSearchCV ({CV_FOLDS}-fold, {len(grid_search.param_grid)} params) ...")
    # Use a subset for speed during tuning; remove [:500] for full tuning
    grid_search.fit(X_train[:500], y_train[:500])

    best_model = grid_search.best_estimator_
    print(f"[train] Best parameters: {grid_search.best_params_}")

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    y_pred = best_model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n[train] ── Results ──────────────────────────────")
    print(f"[train] Test Accuracy : {acc * 100:.2f}%")
    print(f"\n[train] Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return best_model, le, pca


if __name__ == "__main__":
    train()
