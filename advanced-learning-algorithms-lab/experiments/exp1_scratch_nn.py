"""Experiment 1 - neural networks from scratch (NumPy): binary + multiclass."""
from alg_lab.data import load_binary, load_multiclass
from alg_lab.nn_scratch import NeuralNetwork

from .common import out_path, plot_history


def run():
    print("\n=== Experiment 1: neural network from scratch (NumPy) ===")
    results = {}

    # ---- binary classification (sigmoid output)
    d = load_binary()
    net = NeuralNetwork([d.n_features, 16, 8, 1], "sigmoid", l2=1e-3, seed=0)
    net.fit(d.X_train, d.y_train, epochs=150, batch_size=32, lr=1e-3,
            X_val=d.X_cv, y_val=d.y_cv, verbose=50)
    acc = 1 - net.error(d.X_test, d.y_test)
    print(f"[binary]     breast cancer   test accuracy = {acc:.4f}")
    plot_history(net.history, "Scratch NN - breast cancer (sigmoid)", out_path("01_binary_training.png"))
    results["scratch_nn_binary_test_acc"] = acc

    # ---- multiclass classification (softmax output)
    d = load_multiclass()
    net = NeuralNetwork([d.n_features, 64, 32, d.n_classes], "softmax", l2=1e-4, seed=0)
    net.fit(d.X_train, d.y_train, epochs=60, batch_size=32, lr=1e-3,
            X_val=d.X_cv, y_val=d.y_cv, verbose=20)
    acc = 1 - net.error(d.X_test, d.y_test)
    print(f"[multiclass] digits (10-way) test accuracy = {acc:.4f}")
    plot_history(net.history, "Scratch NN - digits (softmax)", out_path("01_multiclass_training.png"))
    results["scratch_nn_multiclass_test_acc"] = acc
    return results


if __name__ == "__main__":
    run()
