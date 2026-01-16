"""Learning goal: evaluate a model more reliably with k-fold cross-validation.

A single train/test split gives ONE estimate that depends heavily on which
samples landed in the test set. k-fold CV rotates the held-out fold so every
sample is tested exactly once, then averages the scores. The mean is a more
stable estimate and the std tells us how much the score wobbles.
"""

import random

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Reproducibility first.
random.seed(0)
np.random.seed(0)


def build_model():
    """A simple, well-behaved classifier suitable for the Iris dataset."""
    # max_iter bumped up so the solver comfortably converges.
    return LogisticRegression(max_iter=1000, random_state=0)


def main():
    # load_iris is a built-in, offline dataset (150 samples, 3 classes).
    X, y = load_iris(return_X_y=True)

    model = build_model()

    # StratifiedKFold preserves the class proportions inside each fold, which is
    # important so no fold accidentally misses a class. shuffle=True randomizes
    # the assignment (seeded for reproducibility).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # cross_val_score trains and evaluates the model once per fold and returns
    # one score per fold.
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    print("Per-fold accuracy scores:")
    for i, s in enumerate(scores, start=1):
        print(f"  Fold {i}: {s:.3f}")

    # Report the central tendency and the spread together. The std quantifies
    # how sensitive the estimate is to the particular data partition.
    print(f"\nMean accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    # Why CV beats a single split:
    # - Every sample is used for both training and testing (across folds), so we
    #   use the data efficiently and don't waste a fixed chunk on one test set.
    # - Averaging over folds reduces the variance of the estimate, so a lucky or
    #   unlucky split can't dominate our conclusion.
    # - The std gives a confidence sense, a single split offers no such signal.
    print(
        "\nCV is more reliable than one split: it averages over multiple held-out"
        "\nfolds, so the estimate doesn't hinge on a single (possibly lucky) split."
    )


if __name__ == "__main__":
    main()
