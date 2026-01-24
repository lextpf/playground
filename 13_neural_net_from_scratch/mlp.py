"""A minimal Multi-Layer Perceptron with one hidden layer.

Learning goal: derive and implement forward pass + backpropagation + plain
gradient descent by hand, reusing the activations from activations.py. The
hidden nonlinearity is what lets this model carve nonlinear decision
boundaries (e.g. XOR), unlike the single perceptron.

Architecture: input -> hidden (tanh) -> output (sigmoid), trained with binary
cross-entropy. The cross-entropy + sigmoid pairing gives the clean output
gradient (a - y), which keeps the backward pass simple and stable.
"""

from __future__ import annotations

import numpy as np

from activations import tanh, tanh_derivative, sigmoid


class MLP:
    """One-hidden-layer network for binary classification."""

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        lr: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.lr = lr
        rng = np.random.default_rng(seed)
        # Small random init breaks symmetry; scale keeps tanh in its active range.
        self.W1: np.ndarray = rng.normal(scale=0.5, size=(n_inputs, n_hidden))
        self.b1: np.ndarray = np.zeros((1, n_hidden))
        self.W2: np.ndarray = rng.normal(scale=0.5, size=(n_hidden, 1))
        self.b2: np.ndarray = np.zeros((1, 1))

    def forward(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Run the forward pass, caching pre/post activations for backprop."""
        z1 = X @ self.W1 + self.b1
        a1 = tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid(z2)  # network output (predicted probability)
        return {"z1": z1, "a1": a1, "z2": z2, "a2": a2}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Threshold the output probability at 0.5 to get a 0/1 label."""
        return (self.forward(X)["a2"] >= 0.5).astype(int)

    @staticmethod
    def _bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary cross-entropy with clipping to avoid log(0)."""
        eps = 1e-12
        p = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def _backward(self, X: np.ndarray, y: np.ndarray, cache: dict[str, np.ndarray]) -> None:
        """Backpropagate gradients and apply one gradient-descent step."""
        n = X.shape[0]
        # Output layer: with sigmoid + BCE, dL/dz2 collapses to (a2 - y).
        dz2 = cache["a2"] - y                      # (n, 1)
        dW2 = cache["a1"].T @ dz2 / n              # (hidden, 1)
        db2 = np.sum(dz2, axis=0, keepdims=True) / n
        # Hidden layer: propagate error back through W2 and tanh'.
        dz1 = (dz2 @ self.W2.T) * tanh_derivative(cache["z1"])  # (n, hidden)
        dW1 = X.T @ dz1 / n
        db1 = np.sum(dz1, axis=0, keepdims=True) / n
        # Descend.
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 2000) -> list[float]:
        """Train with full-batch gradient descent; return the loss history."""
        y = y.reshape(-1, 1).astype(float)
        history: list[float] = []
        for _ in range(epochs):
            cache = self.forward(X)
            history.append(self._bce_loss(y, cache["a2"]))
            self._backward(X, y, cache)
        return history


def main() -> None:
    """Smoke test: learn a trivial linearly-shaped rule (output = first bit)."""
    np.random.seed(0)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1], dtype=int)  # label == first feature

    model = MLP(n_inputs=2, n_hidden=4, lr=0.5, seed=0)
    history = model.fit(X, y, epochs=1500)
    preds = model.predict(X).ravel()

    print("MLP smoke test (label == first input bit)")
    print(f"  start loss={history[0]:.4f}  final loss={history[-1]:.4f}")
    print(f"  predictions={preds}  targets={y}")
    assert history[-1] < history[0], "loss did not decrease"
    assert np.array_equal(preds, y), "smoke-test classification incorrect"
    print("  smoke test passed")


if __name__ == "__main__":
    main()
