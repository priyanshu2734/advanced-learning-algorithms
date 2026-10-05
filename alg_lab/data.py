"""Datasets and the train / cross-validation / test split used across all experiments.

Every dataset ships with scikit-learn or is generated locally, so the project
runs fully offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_cv: np.ndarray
    y_cv: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list = field(default_factory=list)
    class_names: list = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_classes(self) -> int:
        return int(max(self.y_train.max(), self.y_cv.max(), self.y_test.max()) + 1)


def three_way_split(X, y, scale=True, seed=42, feature_names=None, class_names=None) -> Split:
    """60 % train / 20 % cross-validation / 20 % test (stratified).

    The scaler is fitted on the training set only, so no information from the
    cv / test sets leaks into training.
    """
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_cv, X_te, y_cv, y_te = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
    )
    if scale:
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_cv, X_te = scaler.transform(X_tr), scaler.transform(X_cv), scaler.transform(X_te)
    return Split(X_tr, y_tr, X_cv, y_cv, X_te, y_te,
                 list(feature_names if feature_names is not None else []),
                 list(class_names if class_names is not None else []))


def load_binary(scale=True, seed=42) -> Split:
    """Breast-cancer diagnosis: 30 numeric features, malignant (0) vs benign (1)."""
    d = load_breast_cancer()
    return three_way_split(d.data, d.target, scale, seed, d.feature_names, d.target_names)


def load_multiclass(seed=42) -> Split:
    """Handwritten digits 0-9: 8x8 images flattened to 64 features in [0, 1]."""
    d = load_digits()
    return three_way_split(d.data / 16.0, d.target, scale=False, seed=seed,
                           feature_names=[f"px{i}" for i in range(64)],
                           class_names=[str(i) for i in range(10)])


def load_skewed(n_samples=20000, positive_rate=0.02, seed=42) -> Split:
    """Synthetic fraud-style dataset where only ~2 % of the examples are positive."""
    X, y = make_classification(
        n_samples=n_samples, n_features=20, n_informative=8, n_redundant=4,
        weights=[1 - positive_rate, positive_rate], class_sep=0.9, flip_y=0.01,
        random_state=seed,
    )
    return three_way_split(X, y, True, seed, [f"f{i}" for i in range(20)], ["normal", "fraud"])
