"""Learning goal: see why a RandomForestClassifier usually beats a single
DecisionTree on the SAME data. A forest = many de-correlated trees (bagging +
random feature subsets) averaged together, which cuts variance without adding
much bias. So the test accuracy should be more stable and typically higher.
"""

import random

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

random.seed(0)
np.random.seed(0)
RANDOM_STATE = 0


def load_data():
    """Same generator/params as decision_tree.py so the comparison is fair."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        n_clusters_per_class=2,
        flip_y=0.05,
        random_state=RANDOM_STATE,
    )
    return train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)


def evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return train_acc, test_acc


def main():
    X_train, X_test, y_train, y_test = load_data()

    # Single fully-grown tree: low bias, HIGH variance (sensitive to the exact
    # training sample -> overfits the noise).
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    tree_train, tree_test = evaluate(tree, X_train, X_test, y_train, y_test)

    # Random forest: 200 trees, each on a bootstrap sample + random feature
    # subset at every split. Averaging their votes smooths out individual quirks.
    forest = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",  # the key knob that de-correlates the trees
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    forest_train, forest_test = evaluate(forest, X_train, X_test, y_train, y_test)

    print("Model comparison (same train/test split)")
    print("                       | train_acc | test_acc")
    print("-----------------------+-----------+---------")
    print(f"DecisionTree (depth=*) | {tree_train:>9.3f} | {tree_test:>8.3f}")
    print(f"RandomForest (n=200)   | {forest_train:>9.3f} | {forest_test:>8.3f}")
    print()

    delta = forest_test - tree_test
    print(f"Test-accuracy improvement from ensembling: {delta:+.3f}")
    print()
    print("Why ensembling reduces variance:")
    print("  A single deep tree fits the training set almost perfectly but its")
    print("  decisions are unstable, small data changes flip many splits.")
    print("  Each forest tree sees a bootstrap sample and only sqrt(n_features)")
    print("  candidates per split, so the trees make *different* errors. Averaging")
    print("  many roughly-unbiased, weakly-correlated predictors keeps the bias")
    print("  but shrinks variance ~ (1/T) of the per-tree variance -> better, more")
    print("  reliable test accuracy.")


if __name__ == "__main__":
    main()
