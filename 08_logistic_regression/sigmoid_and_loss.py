"""Learning goal: build reusable, numerically-stable building blocks for
logistic regression, the sigmoid activation and binary cross-entropy loss.

Keeping these in their own module so the training script can just import them.
The interesting bit here is *numerical stability*: a naive sigmoid overflows
for large negative inputs, and BCE blows up to inf when a probability hits
exactly 0 or 1. Both are handled below.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic sigmoid, elementwise.

    The naive form 1 / (1 + exp(-z)) overflows exp() for very negative z.
    Trick: use exp(z) / (1 + exp(z)) where z < 0 so we only ever exp a
    non-positive number. Mathematically identical, just won't warn/overflow.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out


def binary_cross_entropy(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12
) -> float:
    """Mean binary cross-entropy loss.

    y_pred are probabilities in (0, 1). We clip into [eps, 1 - eps] first so
    log(0) -> -inf never happens (a prediction of exactly 0 or 1 that's wrong
    would otherwise give infinite loss).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), eps, 1.0 - eps)
    losses = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return float(np.mean(losses))


def main() -> None:
    # Sanity-eyeball the sigmoid over a small symmetric range.
    zs = np.linspace(-6, 6, 7)
    print("sigmoid over a small range:")
    for z, s in zip(zs, sigmoid(zs)):
        print(f"  sigmoid({z:+.1f}) = {s:.4f}")

    # Toy example: 4 labels, plus a "good" and a "bad" set of predictions.
    y = np.array([0, 0, 1, 1])
    good_pred = np.array([0.05, 0.10, 0.90, 0.95])  # confident & correct
    bad_pred = np.array([0.90, 0.85, 0.10, 0.05])  # confident & wrong

    good_loss = binary_cross_entropy(y, good_pred)
    bad_loss = binary_cross_entropy(y, bad_pred)
    print(f"\nBCE on good predictions: {good_loss:.4f}")
    print(f"BCE on bad  predictions: {bad_loss:.4f}")

    # The whole point of the loss: correct predictions must score lower.
    assert good_loss < bad_loss, "loss should be lower for correct predictions"
    print("loss is lower for correct predictions -> OK")


if __name__ == "__main__":
    main()
