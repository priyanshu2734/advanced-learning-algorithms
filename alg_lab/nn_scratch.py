"""A fully-connected neural network written in plain NumPy.

Covers the Week 1-2 material: neurons and layers, forward propagation,
vectorised matrix multiplication, ReLU / sigmoid / softmax activations,
binary and multiclass cross-entropy, back-propagation, mini-batch training
with the Adam optimiser, and L2 regularisation.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def relu(z):
    return np.maximum(0.0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class NeuralNetwork:
    """Dense network: ReLU hidden layers + sigmoid / softmax / linear output.

    Parameters
    ----------
    layer_sizes : e.g. [30, 16, 8, 1] -> 30 inputs, hidden layers of 16 and 8, 1 output
    output      : "sigmoid" (binary), "softmax" (multiclass) or "linear" (regression)
    l2          : regularisation strength lambda (0 disables it)
    """

    def __init__(self, layer_sizes, output="sigmoid", l2=0.0, seed=0):
        if output not in {"sigmoid", "softmax", "linear"}:
            raise ValueError("output must be 'sigmoid', 'softmax' or 'linear'")
        self.sizes = list(layer_sizes)
        self.output = output
        self.l2 = l2
        rng = np.random.default_rng(seed)
        # He initialisation suits ReLU layers
        self.W = [rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out))
                  for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:])]
        self.b = [np.zeros((1, n_out)) for n_out in self.sizes[1:]]
        self.history: dict = {}

    # ---------------------------------------------------------------- forward
    def _forward(self, X):
        """Forward propagation. Returns the activations of every layer."""
        activations = [X]
        A = X
        last = len(self.W) - 1
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A @ W + b                      # matrix multiplication for the whole batch
            if l < last:
                A = relu(Z)
            elif self.output == "sigmoid":
                A = sigmoid(Z)
            elif self.output == "softmax":
                A = softmax(Z)
            else:
                A = Z
            activations.append(A)
        return activations

    def predict_proba(self, X):
        out = self._forward(np.asarray(X, dtype=float))[-1]
        return out.ravel() if self.output == "sigmoid" else out

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)
        if self.output == "sigmoid":
            return (p >= threshold).astype(int)
        if self.output == "softmax":
            return p.argmax(axis=1)
        return p

    # ------------------------------------------------------------------- loss
    def _encode(self, y):
        y = np.asarray(y)
        if self.output == "softmax":
            return np.eye(self.sizes[-1])[y.astype(int)]
        return y.reshape(-1, 1).astype(float)

    def _data_loss(self, A, Y):
        if self.output == "sigmoid":
            A = np.clip(A, EPS, 1 - EPS)
            return float(-np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A)))
        if self.output == "softmax":
            return float(-np.mean(np.sum(Y * np.log(np.clip(A, EPS, 1.0)), axis=1)))
        return float(0.5 * np.mean((A - Y) ** 2))

    def loss(self, X, y):
        """Cost without the regularisation term (so train / cv curves are comparable)."""
        return self._data_loss(self._forward(np.asarray(X, float))[-1], self._encode(y))

    def _total_loss(self, X, Y):
        reg = 0.5 * self.l2 * sum(float(np.sum(W ** 2)) for W in self.W)
        return self._data_loss(self._forward(X)[-1], Y) + reg

    def error(self, X, y):
        """Misclassification rate (1 - accuracy)."""
        return float(np.mean(self.predict(X) != np.asarray(y)))

    # --------------------------------------------------------------- backprop
    def _gradients(self, X, Y):
        """Back-propagation. For sigmoid+BCE and softmax+CCE, dL/dZ = A - Y."""
        acts = self._forward(X)
        m = X.shape[0]
        dZ = (acts[-1] - Y) / m
        dW, db = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            dW[l] = acts[l].T @ dZ + self.l2 * self.W[l]
            db[l] = dZ.sum(axis=0, keepdims=True)
            if l > 0:
                dZ = (dZ @ self.W[l].T) * (acts[l] > 0)   # ReLU derivative
        return dW, db

    # --------------------------------------------------------------- training
    def fit(self, X, y, epochs=100, batch_size=32, lr=1e-3, X_val=None, y_val=None,
            beta1=0.9, beta2=0.999, seed=0, verbose=0):
        """Mini-batch training with Adam. Records loss / error per epoch in self.history."""
        X = np.asarray(X, float)
        Y = self._encode(y)
        rng = np.random.default_rng(seed)
        params = self.W + self.b
        m_t = [np.zeros_like(p) for p in params]
        v_t = [np.zeros_like(p) for p in params]
        t = 0
        hist = {"loss": [], "error": []}
        if X_val is not None:
            hist.update(val_loss=[], val_error=[])

        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = order[start:start + batch_size]
                dW, db = self._gradients(X[idx], Y[idx])
                t += 1
                for i, (p, g) in enumerate(zip(params, dW + db)):
                    m_t[i] = beta1 * m_t[i] + (1 - beta1) * g
                    v_t[i] = beta2 * v_t[i] + (1 - beta2) * g ** 2
                    m_hat = m_t[i] / (1 - beta1 ** t)
                    v_hat = v_t[i] / (1 - beta2 ** t)
                    p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)   # in-place: updates self.W / self.b

            hist["loss"].append(self.loss(X, y))
            hist["error"].append(self.error(X, y))
            if X_val is not None:
                hist["val_loss"].append(self.loss(X_val, y_val))
                hist["val_error"].append(self.error(X_val, y_val))
            if verbose and epoch % verbose == 0:
                msg = f"epoch {epoch:4d}  loss {hist['loss'][-1]:.4f}  err {hist['error'][-1]:.4f}"
                if X_val is not None:
                    msg += f"  val_loss {hist['val_loss'][-1]:.4f}  val_err {hist['val_error'][-1]:.4f}"
                print(msg)
        self.history = hist
        return hist
