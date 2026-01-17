"""Learning goal: understand how a single DecisionTreeClassifier learns, and how
max_depth trades off bias vs variance, i.e. watch a deep tree memorize the
training set (train acc -> 1.0) while test accuracy stalls or drops (overfitting).
"""

import random

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Reproducibility: seed everything we touch.
random.seed(0)
np.random.seed(0)
RANDOM_STATE = 0


def load_data():
    """Synthetic 2-class problem with some noise so overfitting is visible."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        n_clusters_per_class=2,
        flip_y=0.05,  # label noise -> deep trees will chase it
        random_state=RANDOM_STATE,
    )
    return train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)


def fit_tree(X_train, y_train, max_depth=None):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    return clf


def depth_sweep(X_train, X_test, y_train, y_test, depths):
    """Return (depth, train_acc, test_acc) rows across a range of depths."""
    rows = []
    for d in depths:
        clf = fit_tree(X_train, y_train, max_depth=d)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        rows.append((d, train_acc, test_acc))
    return rows


def main():
    X_train, X_test, y_train, y_test = load_data()

    # Baseline: a fully-grown tree (no depth cap). Classic overfitter.
    full_tree = fit_tree(X_train, y_train, max_depth=None)
    full_train = accuracy_score(y_train, full_tree.predict(X_train))
    full_test = accuracy_score(y_test, full_tree.predict(X_test))
    print("Unconstrained DecisionTree (max_depth=None):")
    print(f"  train acc = {full_train:.3f}  (effectively memorizes the data)")
    print(f"  test  acc = {full_test:.3f}")
    print(f"  actual depth reached = {full_tree.get_depth()}")
    print()

    # Depth-vs-accuracy table: the gap between train and test is the overfit signal.
    depths = [1, 2, 3, 5, 8, 12, 20]
    rows = depth_sweep(X_train, X_test, y_train, y_test, depths)

    print("max_depth | train_acc | test_acc |  gap")
    print("----------+-----------+----------+------")
    for d, tr, te in rows:
        print(f"{d:>9} | {tr:>9.3f} | {te:>8.3f} | {tr - te:>5.3f}")
    print()

    # Pick the depth with the best test accuracy as a cheap "sweet spot".
    best_d, best_tr, best_te = max(rows, key=lambda r: r[2])
    print(f"Best test acc at max_depth={best_d}: {best_te:.3f} "
          f"(train {best_tr:.3f}).")
    print("Takeaway: shallow trees underfit (low train AND test); as depth grows "
          "train acc climbs toward 1.0 but test acc plateaus then dips, the "
          "widening train/test gap IS the overfitting.")


if __name__ == "__main__":
    main()
