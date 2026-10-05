"""Experiment 2 - the same problems in TensorFlow / Keras (linear output + from_logits loss)."""
from alg_lab.data import load_binary, load_multiclass

from .common import out_path, plot_history


def run():
    print("\n=== Experiment 2: TensorFlow / Keras ===")
    try:
        from alg_lab import tf_models as tfm
        tfm._tf()
    except ImportError:
        print("TensorFlow is not installed - skipping. Install it with: pip install tensorflow")
        return {}

    results = {}

    d = load_binary()
    model = tfm.build_binary_model(d.n_features)
    h = model.fit(d.X_train, d.y_train, validation_data=(d.X_cv, d.y_cv),
                  epochs=150, batch_size=32, verbose=0)
    _, acc = model.evaluate(d.X_test, d.y_test, verbose=0)
    print(f"[binary]     breast cancer   test accuracy = {acc:.4f}")
    hist = {"loss": h.history["loss"], "val_loss": h.history["val_loss"],
            "error": [1 - a for a in h.history["accuracy"]],
            "val_error": [1 - a for a in h.history["val_accuracy"]]}
    plot_history(hist, "TensorFlow - breast cancer", out_path("02_tf_binary_training.png"))
    results["tf_binary_test_acc"] = float(acc)

    d = load_multiclass()
    model = tfm.build_multiclass_model(d.n_features, d.n_classes)
    h = model.fit(d.X_train, d.y_train, validation_data=(d.X_cv, d.y_cv),
                  epochs=60, batch_size=32, verbose=0)
    _, acc = model.evaluate(d.X_test, d.y_test, verbose=0)
    print(f"[multiclass] digits (10-way) test accuracy = {acc:.4f}")
    hist = {"loss": h.history["loss"], "val_loss": h.history["val_loss"],
            "error": [1 - a for a in h.history["accuracy"]],
            "val_error": [1 - a for a in h.history["val_accuracy"]]}
    plot_history(hist, "TensorFlow - digits (softmax)", out_path("02_tf_multiclass_training.png"))
    results["tf_multiclass_test_acc"] = float(acc)
    return results


if __name__ == "__main__":
    run()
