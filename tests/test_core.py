"""Run with:  pytest -q"""
import numpy as np

from alg_lab.data import load_binary
from alg_lab.diagnostics import diagnose
from alg_lab.nn_scratch import NeuralNetwork, softmax
from alg_lab.tree_scratch import DecisionTree, RandomForest, entropy, information_gain


def test_softmax_rows_sum_to_one():
    z = np.random.default_rng(0).normal(size=(5, 4)) * 50
    assert np.allclose(softmax(z).sum(axis=1), 1.0)


def _grad_check(output, n_out):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 4))
    y = rng.integers(0, n_out, 6) if output == "softmax" else rng.integers(0, 2, 6)
    net = NeuralNetwork([4, 5, 3, n_out], output, l2=0.1, seed=2)
    Y = net._encode(y)
    dW, _ = net._gradients(X, Y)
    eps = 1e-6
    for l in range(len(net.W)):
        rows, cols = net.W[l].shape
        for (i, j) in [(0, 0), (rows - 1, cols - 1)]:
            old = net.W[l][i, j]
            net.W[l][i, j] = old + eps
            up = net._total_loss(X, Y)
            net.W[l][i, j] = old - eps
            down = net._total_loss(X, Y)
            net.W[l][i, j] = old
            assert abs((up - down) / (2 * eps) - dW[l][i, j]) < 1e-6


def test_backprop_matches_numerical_gradient_sigmoid():
    _grad_check("sigmoid", 1)


def test_backprop_matches_numerical_gradient_softmax():
    _grad_check("softmax", 3)


def test_network_learns_breast_cancer():
    d = load_binary()
    net = NeuralNetwork([d.n_features, 16, 8, 1], "sigmoid", l2=1e-3, seed=0)
    net.fit(d.X_train, d.y_train, epochs=60, batch_size=32, lr=1e-3)
    assert 1 - net.error(d.X_test, d.y_test) > 0.93


def test_entropy_values():
    assert entropy([0, 0, 0, 0]) == 0.0
    assert abs(entropy([0, 1, 0, 1]) - 1.0) < 1e-12


def test_information_gain_matches_lecture_example():
    y = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
    ears = np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 0]) == 1
    face = np.array([1, 0, 1, 0, 1, 1, 0, 1, 1, 1]) == 1
    whiskers = np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0]) == 1
    assert abs(information_gain(y, ears) - 0.28) < 0.01
    assert abs(information_gain(y, face) - 0.03) < 0.01
    assert abs(information_gain(y, whiskers) - 0.12) < 0.01


def test_tree_and_forest_fit_and_beat_chance():
    d = load_binary(scale=False)
    tree = DecisionTree(max_depth=4).fit(d.X_train, d.y_train)
    forest = RandomForest(n_trees=10, max_depth=6).fit(d.X_train, d.y_train)
    assert np.mean(tree.predict(d.X_test) == d.y_test) > 0.88
    assert np.mean(forest.predict(d.X_test) == d.y_test) > 0.90


def test_diagnose_labels():
    assert "HIGH VARIANCE" in diagnose(0.01, 0.10, 0.01)
    assert "HIGH BIAS" in diagnose(0.15, 0.16, 0.01)
