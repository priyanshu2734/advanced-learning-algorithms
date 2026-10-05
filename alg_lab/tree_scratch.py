"""Decision tree and random forest written from scratch with NumPy.

Covers Week 4: measuring purity (entropy), choosing a split by information gain,
continuous-valued features, one-hot encoded features (0/1 columns are just a
special case of a threshold split), stopping criteria, sampling with
replacement and the random-forest algorithm.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ purity
def entropy(y, n_classes=None) -> float:
    """H(p) = -sum p_k log2 p_k  (0 for a pure node, 1 for a 50/50 binary node)."""
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=n_classes or 0)
    p = counts[counts > 0] / len(y)
    return float(-(p * np.log2(p)).sum())


def information_gain(y, left_mask) -> float:
    """Entropy at the node minus the weighted entropy of the two branches."""
    y = np.asarray(y, dtype=int)
    left_mask = np.asarray(left_mask, dtype=bool)
    n, n_left = len(y), int(left_mask.sum())
    if n_left in (0, n):
        return 0.0
    w_left = n_left / n
    return entropy(y) - (w_left * entropy(y[left_mask]) + (1 - w_left) * entropy(y[~left_mask]))


def _entropy_rows(counts, totals):
    p = counts / totals[:, None]
    safe = np.where(p > 0, p, 1.0)
    return -(p * np.log2(safe)).sum(axis=1)


def _best_split(X, y_onehot, y, features, min_samples_leaf):
    """Vectorised search over every feature and every midpoint threshold."""
    n = len(y)
    parent = entropy(y, y_onehot.shape[1])
    best_gain, best_feat, best_thr = 0.0, None, None
    for f in features:
        order = np.argsort(X[:, f], kind="stable")
        xs = X[order, f]
        cum = np.cumsum(y_onehot[order], axis=0)
        total = cum[-1]
        cut = np.nonzero(xs[1:] != xs[:-1])[0]          # candidate split after position i
        if len(cut) == 0:
            continue
        n_left = cut + 1
        n_right = n - n_left
        ok = (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
        if not ok.any():
            continue
        cut, n_left, n_right = cut[ok], n_left[ok], n_right[ok]
        left, right = cum[cut], total - cum[cut]
        child = (n_left / n) * _entropy_rows(left, n_left) + (n_right / n) * _entropy_rows(right, n_right)
        gains = parent - child
        i = int(np.argmax(gains))
        if gains[i] > best_gain:
            best_gain, best_feat = float(gains[i]), int(f)
            best_thr = float((xs[cut[i]] + xs[cut[i] + 1]) / 2.0)
    return best_gain, best_feat, best_thr


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "counts", "gain", "n")

    def __init__(self, counts, n):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.counts = counts
        self.gain = 0.0
        self.n = n

    @property
    def is_leaf(self):
        return self.feature is None


class DecisionTree:
    """Binary decision tree using entropy / information gain.

    Stopping criteria (all from the lectures): maximum depth, too few examples
    to split, node already pure, or information gain below `min_gain`.
    """

    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1,
                 min_gain=1e-7, max_features=None, seed=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.max_features = max_features   # None, "sqrt" or an int (used by the forest)
        self.seed = seed

    def fit(self, X, y, n_classes=None):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        self.n_classes_ = n_classes or int(y.max()) + 1
        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.seed)
        self._onehot = np.eye(self.n_classes_)[y]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.root_ = self._build(X, y, np.arange(len(y)), depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        del self._onehot
        return self

    def _n_candidate_features(self):
        if self.max_features is None:
            return self.n_features_
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features_)))
        return min(int(self.max_features), self.n_features_)

    def _build(self, X, y, idx, depth):
        counts = np.bincount(y[idx], minlength=self.n_classes_)
        node = _Node(counts, len(idx))
        if (depth >= self.max_depth or len(idx) < self.min_samples_split
                or np.count_nonzero(counts) == 1):
            return node
        feats = self._rng.choice(self.n_features_, self._n_candidate_features(), replace=False)
        gain, feat, thr = _best_split(X[idx], self._onehot[idx], y[idx], feats, self.min_samples_leaf)
        if feat is None or gain < self.min_gain:
            return node
        go_left = X[idx, feat] <= thr
        node.feature, node.threshold, node.gain = feat, thr, gain
        self.feature_importances_[feat] += gain * len(idx)
        node.left = self._build(X, y, idx[go_left], depth + 1)
        node.right = self._build(X, y, idx[~go_left], depth + 1)
        return node

    def _leaf(self, x):
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict_proba(self, X):
        X = np.asarray(X, float)
        out = np.empty((len(X), self.n_classes_))
        for i, x in enumerate(X):
            c = self._leaf(x).counts
            out[i] = c / c.sum()
        return out

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def depth(self):
        def d(n):
            return 0 if n.is_leaf else 1 + max(d(n.left), d(n.right))
        return d(self.root_)

    def export_text(self, feature_names=None, class_names=None):
        """Readable if/else rendering of the tree."""
        lines = []

        def name(f):
            return feature_names[f] if feature_names is not None and len(feature_names) else f"x[{f}]"

        def walk(node, indent):
            pad = "|   " * indent
            if node.is_leaf:
                k = int(node.counts.argmax())
                label = class_names[k] if class_names is not None and len(class_names) else k
                lines.append(f"{pad}--> predict {label}  (samples={node.n}, counts={node.counts.tolist()})")
                return
            lines.append(f"{pad}if {name(node.feature)} <= {node.threshold:.3f}  (gain={node.gain:.3f})")
            walk(node.left, indent + 1)
            lines.append(f"{pad}else")
            walk(node.right, indent + 1)

        walk(self.root_, 0)
        return "\n".join(lines)


class RandomForest:
    """Bagging + random feature subsets: B trees, each trained on a bootstrap sample
    (sampling with replacement) and a random subset of sqrt(n) features per split."""

    def __init__(self, n_trees=25, max_depth=8, max_features="sqrt", min_samples_leaf=1, seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        self.n_classes_ = int(y.max()) + 1
        rng = np.random.default_rng(self.seed)
        self.trees_ = []
        for b in range(self.n_trees):
            sample = rng.integers(0, len(X), len(X))        # sampling with replacement
            tree = DecisionTree(max_depth=self.max_depth, max_features=self.max_features,
                                min_samples_leaf=self.min_samples_leaf,
                                seed=int(rng.integers(1 << 31)))
            tree.fit(X[sample], y[sample], n_classes=self.n_classes_)
            self.trees_.append(tree)
        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees_], axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
