"""Pipelines: chain StandardScaler + classifier and cross-validate cleanly.

Learning goal: understand why wrapping preprocessing and the estimator in a
single Pipeline is the idiomatic sklearn pattern, it ties them together so
cross-validation refits the scaler on each training fold, avoiding leakage.
"""

import random

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

random.seed(0)
np.random.seed(0)

RANDOM_STATE = 0


def build_pipeline():
    """Compose scaling + classification as one estimator.

    Why this prevents leakage: cross_val_score clones the whole pipeline per
    fold, so StandardScaler computes its mean/std only on that fold's training
    data. If we scaled the full X up front instead, statistics from the
    validation rows would bleed into training and inflate the score.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=RANDOM_STATE)),
        ]
    )


def main():
    X, y = load_iris(return_X_y=True)

    pipe = build_pipeline()

    # 5-fold CV; each fold scales independently inside the pipeline.
    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")

    print(f"CV accuracy per fold: {np.round(scores, 3)}")
    print(f"Mean: {scores.mean():.3f}  Std: {scores.std():.3f}")


if __name__ == "__main__":
    main()
