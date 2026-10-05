"""Tools for the 'Advice for applying machine learning' week.

Bias / variance diagnosis, regularisation sweeps, learning curves,
and metrics for skewed datasets (precision, recall, F1, threshold trade-off).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from .nn_scratch import NeuralNetwork


def diagnose(train_err, cv_err, baseline_err=0.0, tol=0.02) -> str:
    """Compare train error, cv error and a baseline (e.g. human-level) error.

    bias gap     = train_err - baseline_err   (can the model even fit the training set?)
    variance gap = cv_err   - train_err       (does it generalise?)
    """
    bias_gap = train_err - baseline_err
    var_gap = cv_err - train_err
    high_bias, high_var = bias_gap > tol, var_gap > tol
    if high_bias and high_var:
        return "HIGH BIAS + HIGH VARIANCE - model underfits and does not generalise"
    if high_bias:
        return "HIGH BIAS (underfitting) - try a bigger network, more features, lower lambda"
    if high_var:
        return "HIGH VARIANCE (overfitting) - try more data, a smaller model, higher lambda"
    return "Looks good - bias and variance are both low"


def regularization_sweep(split, lambdas, hidden=(64, 32), epochs=200, lr=1e-3, seed=0):
    """Train one network per lambda; return train / cv error lists."""
    train_err, cv_err = [], []
    for lam in lambdas:
        net = NeuralNetwork([split.n_features, *hidden, 1], "sigmoid", l2=lam, seed=seed)
        net.fit(split.X_train, split.y_train, epochs=epochs, batch_size=32, lr=lr, seed=seed)
        train_err.append(net.error(split.X_train, split.y_train))
        cv_err.append(net.error(split.X_cv, split.y_cv))
    return train_err, cv_err


def learning_curve(split, fractions, hidden=(16, 8), l2=1e-3, epochs=150, lr=1e-3, seed=0):
    """Train on growing slices of the training set; evaluate on that slice and the full cv set."""
    sizes, train_err, cv_err = [], [], []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(split.y_train))
    for frac in fractions:
        n = max(10, int(frac * len(order)))
        idx = order[:n]
        net = NeuralNetwork([split.n_features, *hidden, 1], "sigmoid", l2=l2, seed=seed)
        net.fit(split.X_train[idx], split.y_train[idx], epochs=epochs, batch_size=32, lr=lr, seed=seed)
        sizes.append(n)
        train_err.append(net.error(split.X_train[idx], split.y_train[idx]))
        cv_err.append(net.error(split.X_cv, split.y_cv))
    return sizes, train_err, cv_err


def threshold_tradeoff(y_true, proba, thresholds=None):
    """Precision / recall / F1 as the decision threshold moves (trading off precision and recall)."""
    thresholds = np.linspace(0.05, 0.95, 19) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        rows.append((float(t), float(p), float(r), float(f)))
    return rows
