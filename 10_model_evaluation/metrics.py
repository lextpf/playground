"""Learning goal: go beyond accuracy with precision, recall, F1, and the
confusion matrix.

Accuracy alone can be misleading (especially with class imbalance). Each metric
answers a different question, so on a held-out set we compute all of them to get
a fuller picture of WHERE and HOW the model is right or wrong.
"""

import random

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Seed all randomness for a deterministic run.
random.seed(0)
np.random.seed(0)


def build_dataset():
    """Imbalanced binary problem so the metrics differ meaningfully."""
    # weights skews the classes (~85% / 15%), this is where accuracy starts to
    # lie and precision/recall earn their keep.
    X, y = make_classification(
        n_samples=600,
        n_features=12,
        n_informative=6,
        n_classes=2,
        weights=[0.85, 0.15],
        flip_y=0.03,
        random_state=0,
    )
    return X, y


def main():
    X, y = build_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    # Predict on the held-out set only, never score on training data here.
    y_pred = clf.predict(X_test)

    # Accuracy: fraction of all predictions that are correct. Easy to read but
    # can look great even when the rare class is mostly missed.
    acc = accuracy_score(y_test, y_pred)
    # Precision: of the samples predicted positive, how many truly are. High
    # precision = few false positives ("when it says yes, trust it").
    prec = precision_score(y_test, y_pred, zero_division=0)
    # Recall: of the actual positives, how many we caught. High recall = few
    # false negatives ("it rarely misses a real positive").
    rec = recall_score(y_test, y_pred, zero_division=0)
    # F1: harmonic mean of precision and recall, a single balanced score that
    # punishes ignoring either one.
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 score:  {f1:.3f}")

    # The confusion matrix breaks predictions into TN/FP/FN/TP, the raw counts
    # every metric above is derived from. Rows = actual class, cols = predicted.
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(cm)
    print(f"  True Negatives : {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives : {tp}")

    # Takeaway: on an imbalanced set, inspect precision/recall and the confusion
    # matrix together, accuracy can stay high while the minority class suffers.


if __name__ == "__main__":
    main()
