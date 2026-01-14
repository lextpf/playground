"""Learning goal: implement logistic regression from scratch with batch
gradient descent, reusing the sigmoid + BCE helpers I wrote earlier.

I'm wrapping the model in a small class this time (intermediate-stage habit)
so weights/bias live together and `fit`/`predict_proba`/`predict` read like
the sklearn API I'm used to. Dataset is a seeded, roughly linearly-separable
2D blob so I can reason about whether the numbers make sense.
"""

from __future__ import annotations

import numpy as np

from sigmoid_and_loss import binary_cross_entropy, sigmoid


def make_dataset(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two Gaussian blobs in 2D, linearly-separable-ish (they overlap a bit).

    Returns X of shape (n, 2) and binary labels y of shape (n,).
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    class0 = rng.normal(loc=[-2.0, -2.0], scale=1.0, size=(half, 2))
    class1 = rng.normal(loc=[2.0, 2.0], scale=1.0, size=(n - half, 2))
    X = np.vstack([class0, class1])
    y = np.concatenate([np.zeros(half), np.ones(n - half)])
    # Shuffle so the two classes aren't perfectly ordered.
    perm = rng.permutation(n)
    return X[perm], y[perm]


class LogisticRegression:
    """Plain logistic regression trained by full-batch gradient descent."""

    def __init__(self, lr: float = 0.1, n_iters: int = 500) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            preds = self.predict_proba(X)
            error = preds - y  # gradient of BCE w.r.t. the logit, nice & clean
            grad_w = (X.T @ error) / n_samples
            grad_b = float(np.mean(error))
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.weights is not None, "call fit() before predicting"
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def main() -> None:
    np.random.seed(0)  # belt-and-suspenders global seed for reproducibility
    X, y = make_dataset(n=200, seed=0)

    model = LogisticRegression(lr=0.1, n_iters=500).fit(X, y)

    final_loss = binary_cross_entropy(y, model.predict_proba(X))
    train_acc = accuracy(y, model.predict(X))

    print(f"learned weights: {model.weights}")
    print(f"learned bias:    {model.bias:.4f}")
    print(f"final BCE loss:  {final_loss:.4f}")
    print(f"train accuracy:  {train_acc:.3f}")


if __name__ == "__main__":
    main()
