"""Evaluation for the digit classifier capstone.

Learning goal: load the trained weights and judge the model on the held-out
test set, overall accuracy plus a per-class breakdown and a confusion matrix.
I lean on scikit-learn's metrics here because they're battle-tested and give a
much richer picture than a single accuracy number.

Run this from INSIDE this folder, after train.py:  python evaluate.py
"""

import os

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from data import load_data, seed_everything
from model import build_model

CKPT_PATH = os.path.join(os.path.dirname(__file__), "digit_cnn.pt")


def load_trained_model(n_classes):
    """Rebuild the architecture and load saved weights from this folder."""
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"No checkpoint at {CKPT_PATH}. Run `python train.py` first."
        )
    model = build_model(n_classes=n_classes)
    # weights_only=True: the checkpoint is just a state_dict (tensors), so this
    # avoids unpickling arbitrary objects, the safe way to load.
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
    model.eval()
    return model


def evaluate():
    """Score the trained model on the test split and print a full report."""
    seed_everything()

    data = load_data()
    model = load_trained_model(data["n_classes"])

    with torch.no_grad():
        logits = model(data["X_test"])
        preds = logits.argmax(dim=1).numpy()

    y_true = data["y_test"].numpy()

    acc = accuracy_score(y_true, preds)
    print(f"overall test accuracy: {acc:.4f}\n")

    # Per-class precision/recall/F1, the recall column is effectively
    # per-class accuracy, which is exactly what I want to inspect.
    print("per-class report:")
    print(classification_report(y_true, preds, digits=3))

    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, preds))

    return acc


def main():
    evaluate()


if __name__ == "__main__":
    main()
