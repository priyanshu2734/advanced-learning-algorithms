"""Experiment 4 - decision trees, random forests and boosted trees (XGBoost)."""
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from alg_lab.data import load_binary, load_multiclass
from alg_lab.tree_scratch import DecisionTree, RandomForest, entropy, information_gain

from .common import out_path, plt


def _lecture_example():
    """The cat-vs-dog example: information gain of each root split."""
    # columns: pointy_ears, round_face, whiskers_present ; label: cat = 1
    X = np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1],
                  [1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0]])
    y = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
    print(f"\n-- cat/dog toy example: root entropy = {entropy(y):.3f}")
    for name, col in zip(["ear shape (pointy)", "face shape (round)", "whiskers (present)"], X.T):
        print(f"   information gain, split on {name:<20} = {information_gain(y, col == 1):.3f}")


def _xgb_or_fallback():
    try:
        from xgboost import XGBClassifier
        return "XGBoost", lambda: XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                                eval_metric="logloss", random_state=0)
    except ImportError:
        print("   (xgboost not installed -> using scikit-learn HistGradientBoosting instead; "
              "pip install xgboost to use the real thing)")
        return "HistGradientBoosting*", lambda: HistGradientBoostingClassifier(max_depth=4, random_state=0)


def _compare(split, title):
    xgb_name, xgb_factory = _xgb_or_fallback()
    models = {
        "Tree (scratch)": lambda: DecisionTree(max_depth=5),
        "Tree (sklearn)": lambda: DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=0),
        "Forest (scratch, 25)": lambda: RandomForest(n_trees=25, max_depth=8),
        "Forest (sklearn, 200)": lambda: RandomForestClassifier(n_estimators=200, random_state=0),
        xgb_name: xgb_factory,
    }
    print(f"\n-- {title}")
    print(f"{'model':<24}{'train acc':>10}{'cv acc':>9}{'test acc':>10}{'time (s)':>10}")
    scores = {}
    for name, make in models.items():
        t0 = time.time()
        m = make().fit(split.X_train, split.y_train)
        dt = time.time() - t0
        tr, cv, te = (float(np.mean(m.predict(X) == y)) for X, y in
                      [(split.X_train, split.y_train), (split.X_cv, split.y_cv), (split.X_test, split.y_test)])
        print(f"{name:<24}{tr:>10.3f}{cv:>9.3f}{te:>10.3f}{dt:>10.2f}")
        scores[name] = te
    return scores


def _depth_curve(split):
    """Deeper tree -> lower bias, higher variance: pick max_depth with the CV set."""
    depths = range(1, 13)
    tr, cv = [], []
    for k in depths:
        m = DecisionTreeClassifier(criterion="entropy", max_depth=k, random_state=0).fit(split.X_train, split.y_train)
        tr.append(1 - m.score(split.X_train, split.y_train))
        cv.append(1 - m.score(split.X_cv, split.y_cv))
    plt.figure(figsize=(6, 4))
    plt.plot(depths, tr, "o-", label="train error")
    plt.plot(depths, cv, "s-", label="cross-validation error")
    plt.xlabel("max_depth")
    plt.ylabel("error")
    plt.title("Decision tree: depth vs error")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path("04_tree_depth_curve.png"), dpi=130)
    plt.close()


def _importances(split):
    rf = RandomForest(n_trees=25, max_depth=8).fit(split.X_train, split.y_train)
    imp = rf.feature_importances_
    top = np.argsort(imp)[::-1][:10][::-1]
    plt.figure(figsize=(7, 4.2))
    plt.barh([split.feature_names[i] for i in top], imp[top])
    plt.title("Random forest (scratch) - top 10 features")
    plt.tight_layout()
    plt.savefig(out_path("04_feature_importance.png"), dpi=130)
    plt.close()


def _comparison_plot(all_scores):
    names = list(next(iter(all_scores.values())).keys())
    width = 0.38
    plt.figure(figsize=(9, 4.2))
    for j, (ds, sc) in enumerate(all_scores.items()):
        plt.bar(np.arange(len(names)) + j * width, [sc[n] for n in names], width, label=ds)
    plt.xticks(np.arange(len(names)) + width / 2, names, rotation=15, ha="right")
    plt.ylim(0.7, 1.0)
    plt.ylabel("test accuracy")
    plt.title("Tree-based models compared")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path("04_model_comparison.png"), dpi=130)
    plt.close()


def run():
    print("\n=== Experiment 4: decision trees, random forests, boosted trees ===")
    _lecture_example()
    b, m = load_binary(scale=False), load_multiclass()

    tree = DecisionTree(max_depth=3).fit(b.X_train, b.y_train)
    print("\n-- scratch tree (depth 3) on breast cancer:")
    print(tree.export_text(b.feature_names, b.class_names))

    scores = {"breast cancer": _compare(b, "breast cancer (binary)"),
              "digits": _compare(m, "digits (10 classes)")}
    _depth_curve(b)
    _importances(b)
    _comparison_plot(scores)
    return {f"trees_{ds.replace(' ', '_')}": sc for ds, sc in scores.items()}


if __name__ == "__main__":
    run()
