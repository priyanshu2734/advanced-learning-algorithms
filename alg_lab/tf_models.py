"""TensorFlow / Keras versions of the networks (Week 1-2 coding practice).

Following the course recommendation, the output layers use *linear* activations
and the loss is built with `from_logits=True`. This is numerically more stable
than putting sigmoid / softmax inside the last layer.
"""
from __future__ import annotations


def _tf():
    try:
        import tensorflow as tf
        from tensorflow import keras
        return tf, keras
    except ImportError as exc:  # pragma: no cover
        raise ImportError("TensorFlow is required for this module: pip install tensorflow") from exc


def build_binary_model(n_features, hidden=(16, 8), l2=1e-3, lr=1e-3, seed=0):
    tf, keras = _tf()
    keras.utils.set_random_seed(seed)
    reg = keras.regularizers.l2(l2)
    model = keras.Sequential(
        [keras.Input(shape=(n_features,))]
        + [keras.layers.Dense(u, activation="relu", kernel_regularizer=reg) for u in hidden]
        + [keras.layers.Dense(1, activation="linear")]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(threshold=0.0, name="accuracy")],  # logit 0 <-> p = 0.5
    )
    return model


def build_multiclass_model(n_features, n_classes, hidden=(64, 32), l2=1e-4, lr=1e-3, seed=0):
    tf, keras = _tf()
    keras.utils.set_random_seed(seed)
    reg = keras.regularizers.l2(l2)
    model = keras.Sequential(
        [keras.Input(shape=(n_features,))]
        + [keras.layers.Dense(u, activation="relu", kernel_regularizer=reg) for u in hidden]
        + [keras.layers.Dense(n_classes, activation="linear")]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def predict_proba_binary(model, X):
    tf, _ = _tf()
    return tf.nn.sigmoid(model.predict(X, verbose=0)).numpy().ravel()


def predict_proba_multiclass(model, X):
    tf, _ = _tf()
    return tf.nn.softmax(model.predict(X, verbose=0)).numpy()
