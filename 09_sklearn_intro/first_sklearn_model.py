"""First scikit-learn model: load Iris, split, train a classifier, evaluate.

Learning goal: practice the canonical sklearn workflow, load data,
train/test split (seeded), fit an estimator, and read off accuracy plus a
per-class classification_report.
"""

import random

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Seed everything we touch so the run is reproducible.
random.seed(0)
np.random.seed(0)

RANDOM_STATE = 0


def load_data():
    """Return (X, y, target_names) from the built-in Iris dataset."""
    iris = load_iris()
    return iris.data, iris.target, iris.target_names


def train_model(X_train, y_train):
    """Fit a LogisticRegression on the training fold."""
    # max_iter bumped so the lbfgs solver comfortably converges on Iris.
    clf = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    return clf


def main():
    X, y, target_names = load_data()

    # stratify keeps class proportions balanced across train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    clf = train_model(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=target_names))


if __name__ == "__main__":
    main()
