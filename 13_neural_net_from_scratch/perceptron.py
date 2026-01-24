"""Single-layer Perceptron trained with the classic perceptron update rule.

Learning goal: implement Rosenblatt's perceptron from scratch and watch it
learn the AND gate. AND is linearly separable, so a single linear boundary
(one weight vector + bias) is enough, this sets up the contrast with XOR,
which a single perceptron provably *cannot* solve.

Update rule (per misclassified sample): w += lr * (y_true - y_pred) * x.
This nudges the decision boundary toward correctly classifying the point.
"""

from __future__ import annotations

import numpy as np


class Perceptron:
    """A single linear threshold unit (step activation)."""

    def __init__(self, n_features: int, lr: float = 0.1, epochs: int = 50) -> None:
        self.lr = lr
        self.epochs = epochs
        # weights initialized to zero; bias kept separate for clarity.
        self.weights: np.ndarray = np.zeros(n_features, dtype=float)
        self.bias: float = 0.0

    def _net_input(self, X: np.ndarray) -> np.ndarray:
        """Linear combination w . x + b."""
        return X @ self.weights + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply the unit step: output 1 if net input >= 0, else 0."""
        return (self._net_input(X) >= 0.0).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """Train via the perceptron rule, iterating until convergence or epochs."""
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                prediction = int(self._net_input(xi) >= 0.0)
                update = self.lr * (target - prediction)
                if update != 0.0:
                    self.weights += update * xi
                    self.bias += update
                    errors += 1
            if errors == 0:  # linearly separable data => exact convergence
                break
        return self


def main() -> None:
    np.random.seed(0)
    # AND gate truth table.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1], dtype=int)

    model = Perceptron(n_features=2, lr=0.1, epochs=50).fit(X, y)
    predictions = model.predict(X)

    print("Perceptron learning the AND gate")
    print(f"weights={model.weights}, bias={model.bias:.2f}")
    for xi, target, pred in zip(X, y, predictions):
        print(f"  input={xi.astype(int)} -> pred={pred}  (target={target})")

    accuracy = float(np.mean(predictions == y))
    assert accuracy == 1.0, "perceptron failed to learn linearly separable AND"
    print(f"All 4 inputs classified correctly (accuracy={accuracy:.0%})")


if __name__ == "__main__":
    main()
