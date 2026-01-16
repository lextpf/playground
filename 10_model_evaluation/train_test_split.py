"""Learning goal: understand why we hold out a test set.

A model can score near-perfectly on data it has already seen, yet fail on new
data. By splitting into train/test and comparing the two accuracies we get a
first, honest look at generalization vs. overfitting.
"""

import random

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Seed every source of randomness so the run is reproducible.
random.seed(0)
np.random.seed(0)


def build_dataset():
    """Create a small, slightly noisy synthetic classification problem."""
    # A few informative features + noise so the task isn't trivially separable.
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        flip_y=0.05,  # label noise -> makes overfitting visible
        random_state=0,
    )
    return X, y


def main():
    X, y = build_dataset()

    # stratify=y keeps the class ratio the same in both splits, which matters
    # for fair evaluation. test_size=0.25 holds out a quarter of the data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )

    # A deep decision tree is intentionally high-variance: it can memorize the
    # training set, which exaggerates the train-vs-test gap for teaching.
    clf = DecisionTreeClassifier(max_depth=None, random_state=0)
    clf.fit(X_train, y_train)

    # Training accuracy: how well the model fits data it learned from.
    train_acc = clf.score(X_train, y_train)
    # Test accuracy: the number we actually care about, performance on UNSEEN
    # data. This estimates how the model behaves in the real world.
    test_acc = clf.score(X_test, y_test)

    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy:     {test_acc:.3f}")
    print(f"Generalization gap: {train_acc - test_acc:.3f}")

    # Interpretation: a large positive gap (train >> test) signals overfitting
    # the model learned noise specific to the training set. We trust the test
    # accuracy, not the training accuracy, when judging the model.
    if train_acc - test_acc > 0.1:
        print("Note: sizeable gap -> the model is overfitting the training data.")
    else:
        print("Note: small gap -> the model generalizes reasonably well.")


if __name__ == "__main__":
    main()
