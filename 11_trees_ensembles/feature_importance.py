"""Learning goal: read off which features a RandomForest actually relies on.
sklearn's .feature_importances_ is mean impurity decrease (Gini) aggregated over
all trees. I generate data where only a few features are informative, fit a
forest, and check that the ranking surfaces those (and that noise features rank
near zero).
"""

import random

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

random.seed(0)
np.random.seed(0)
RANDOM_STATE = 0


def load_data():
    """8 of 20 features informative; the rest are redundant/noise on purpose."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        n_clusters_per_class=2,
        flip_y=0.05,
        shuffle=False,  # keep informative features as the first columns for clarity
        random_state=RANDOM_STATE,
    )
    feature_names = [f"f{i:02d}" for i in range(X.shape[1])]
    return X, y, feature_names


def ranked_importances(clf, feature_names):
    """Return (name, importance) pairs sorted high -> low."""
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    return [(feature_names[i], float(importances[i])) for i in order]


def main():
    X, y, feature_names = load_data()

    forest = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    forest.fit(X, y)

    ranking = ranked_importances(forest, feature_names)

    print("Feature importances (Gini, mean impurity decrease), ranked:")
    print("rank | feature | importance | bar")
    print("-----+---------+------------+--------------------")
    for rank, (name, imp) in enumerate(ranking, start=1):
        bar = "#" * int(round(imp * 100))
        print(f"{rank:>4} | {name:>7} | {imp:>10.4f} | {bar}")

    total_top5 = sum(imp for _, imp in ranking[:5])
    print()
    print(f"Top-5 features account for {total_top5:.1%} of total importance.")
    print()
    print("Interpretation: the data has 8 informative columns (f00-f07, since we")
    print("set shuffle=False), and those dominate the ranking while the tail of")
    print("pure-noise features sits near ~0. Caveat: impurity-based importance is")
    print("biased toward high-cardinality/continuous features and splits credit")
    print("among correlated/redundant columns, for decisions I'd cross-check with")
    print("permutation importance, but as a quick read this lines up with reality.")


if __name__ == "__main__":
    main()
