"""Experiment 3 - advice for applying machine learning.

* bias / variance diagnosis against a baseline
* choosing lambda with the cross-validation set (test set untouched until the end)
* learning curves
* error analysis (confusion matrix + misclassified digits)
* skewed dataset: why accuracy lies, precision / recall trade-off
"""
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support)

from alg_lab.data import load_binary, load_multiclass, load_skewed
from alg_lab.diagnostics import (diagnose, learning_curve, regularization_sweep, threshold_tradeoff)
from alg_lab.nn_scratch import NeuralNetwork

from .common import out_path, plt

BASELINE_ERR = 0.03   # assumed best-achievable error for breast-cancer screening (documented assumption)


def _bias_variance_and_lambda():
    d = load_binary()
    lambdas = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    tr, cv = regularization_sweep(d, lambdas)
    print("\n-- regularisation sweep (network 64-32, 200 epochs)")
    print(f"{'lambda':>8} {'train err':>10} {'cv err':>8}   diagnosis")
    for lam, a, b in zip(lambdas, tr, cv):
        print(f"{lam:>8} {a:>10.4f} {b:>8.4f}   {diagnose(a, b, BASELINE_ERR)}")
    # lowest CV error wins; ties go to the larger lambda (the simpler model)
    best = max(l for l, e in zip(lambdas, cv) if e == min(cv))
    print(f"best lambda on the CV set: {best}")

    x = [max(l, 1e-5) for l in lambdas]
    plt.figure(figsize=(6, 4))
    plt.semilogx(x, tr, "o-", label="train error")
    plt.semilogx(x, cv, "s-", label="cross-validation error")
    plt.axhline(BASELINE_ERR, ls="--", c="gray", label="baseline error")
    plt.xlabel("lambda (0 plotted at 1e-5)")
    plt.ylabel("error")
    plt.title("Bias / variance vs regularisation")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path("03_regularization_sweep.png"), dpi=130)
    plt.close()

    # learning curve with the chosen lambda
    sizes, tr_e, cv_e = learning_curve(d, np.linspace(0.1, 1.0, 8), l2=best)
    plt.figure(figsize=(6, 4))
    plt.plot(sizes, tr_e, "o-", label="train error")
    plt.plot(sizes, cv_e, "s-", label="cross-validation error")
    plt.axhline(BASELINE_ERR, ls="--", c="gray", label="baseline error")
    plt.xlabel("training set size")
    plt.ylabel("error")
    plt.title("Learning curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path("03_learning_curve.png"), dpi=130)
    plt.close()

    net = NeuralNetwork([d.n_features, 64, 32, 1], "sigmoid", l2=best, seed=0)
    net.fit(d.X_train, d.y_train, epochs=200, batch_size=32, lr=1e-3)
    test_err = net.error(d.X_test, d.y_test)
    print(f"final model (lambda={best}) -> test error {test_err:.4f} (test set used once)")
    return {"best_lambda": best, "final_test_error": test_err}


def _error_analysis():
    d = load_multiclass()
    net = NeuralNetwork([d.n_features, 64, 32, d.n_classes], "softmax", l2=1e-4, seed=0)
    net.fit(d.X_train, d.y_train, epochs=60, batch_size=32, lr=1e-3)
    pred = net.predict(d.X_cv)
    wrong = np.nonzero(pred != d.y_cv)[0]
    print(f"\n-- error analysis on digits: {len(wrong)} / {len(d.y_cv)} cv examples wrong")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(d.y_cv, pred)).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (cross-validation set)")
    fig.tight_layout()
    fig.savefig(out_path("03_confusion_matrix.png"), dpi=130)
    plt.close(fig)

    show = wrong[:12]
    if len(show):
        fig, axes = plt.subplots(2, 6, figsize=(10, 3.8))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), show):
            ax.imshow(d.X_cv[i].reshape(8, 8), cmap="gray_r")
            ax.set_title(f"true {d.y_cv[i]} / pred {pred[i]}", fontsize=9)
        fig.suptitle("Misclassified digits - read them to decide what to fix next")
        fig.tight_layout()
        fig.savefig(out_path("03_misclassified_digits.png"), dpi=130)
        plt.close(fig)
    return {"digits_cv_errors": int(len(wrong))}


def _skewed():
    d = load_skewed()
    rate = d.y_test.mean()
    always_zero_acc = 1 - rate
    net = NeuralNetwork([d.n_features, 16, 8, 1], "sigmoid", l2=1e-3, seed=0)
    net.fit(d.X_train, d.y_train, epochs=60, batch_size=128, lr=1e-3)
    proba = net.predict_proba(d.X_test)
    pred = (proba >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(d.y_test, pred, average="binary", zero_division=0)
    print(f"\n-- skewed data ({rate:.1%} positives)")
    print(f"'always predict 0' accuracy = {always_zero_acc:.4f}  <- accuracy alone is misleading")
    print(f"network @0.5: accuracy {np.mean(pred == d.y_test):.4f}  precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}")

    rows = threshold_tradeoff(d.y_test, proba)
    t, pr, rc, f1 = zip(*rows)
    plt.figure(figsize=(6, 4))
    plt.plot(t, pr, "o-", label="precision")
    plt.plot(t, rc, "s-", label="recall")
    plt.plot(t, f1, "^-", label="F1")
    plt.xlabel("decision threshold")
    plt.title("Precision / recall trade-off (skewed data)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path("03_precision_recall_tradeoff.png"), dpi=130)
    plt.close()
    return {"skewed_always_zero_acc": float(always_zero_acc), "skewed_precision": float(p),
            "skewed_recall": float(r), "skewed_f1": float(f)}


def run():
    print("\n=== Experiment 3: advice for applying machine learning ===")
    out = {}
    out.update(_bias_variance_and_lambda())
    out.update(_error_analysis())
    out.update(_skewed())
    return out


if __name__ == "__main__":
    run()
