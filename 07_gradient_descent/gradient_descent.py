"""Learning goal: implement batch gradient descent from scratch to fit a
linear regression with MSE loss.

I want to really understand the mechanics before reaching for sklearn:
- the MSE loss and its gradient w.r.t. the weights,
- the parameter update rule (w <- w - lr * grad),
- and watching the loss history go down over iterations.

I include a bias term by prepending a column of 1s to X (the "bias trick"),
so the bias is just another weight and I don't need special-casing.
"""

import numpy as np


def add_bias_column(X):
    """Prepend a column of 1s so the bias becomes weight[0] (the bias trick)."""
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def mse_loss(X, y, w):
    """Mean squared error for predictions X @ w against targets y.

    I divide by n so the loss (and its gradient) don't grow with dataset size,
    which keeps a chosen learning rate meaningful as n changes.
    """
    residuals = X @ w - y
    return float(np.mean(residuals ** 2))


def mse_gradient(X, y, w):
    """Gradient of the MSE loss w.r.t. w.

    For loss = mean((Xw - y)^2), the gradient is (2/n) * X^T (Xw - y).
    """
    n = X.shape[0]
    residuals = X @ w - y
    return (2.0 / n) * (X.T @ residuals)


def gradient_descent(X, y, lr=0.1, n_iters=1000):
    """Fit linear regression by batch gradient descent.

    Expects X WITHOUT a bias column; we add it internally.
    Returns the learned weight vector (w[0] is the bias) and the loss history
    as a list with one entry per iteration (the loss before each update).
    """
    Xb = add_bias_column(X)
    n_features = Xb.shape[1]
    w = np.zeros(n_features)  # start at the origin; simple and reproducible

    loss_history = []
    for _ in range(n_iters):
        loss_history.append(mse_loss(Xb, y, w))
        grad = mse_gradient(Xb, y, w)
        w = w - lr * grad

    return w, loss_history


def make_synthetic_data(n_samples=200, n_features=3, noise=0.5, seed=0):
    """Generate seeded synthetic data from a known linear model + noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.normal(size=n_features)
    true_bias = 2.0
    y = X @ true_w + true_bias + noise * rng.normal(size=n_samples)
    return X, y, true_w, true_bias


def main():
    np.random.seed(0)  # belt-and-suspenders global seed for reproducibility
    X, y, true_w, true_bias = make_synthetic_data(seed=0)

    w, loss_history = gradient_descent(X, y, lr=0.1, n_iters=1000)

    print("Batch gradient descent for linear regression (MSE)")
    print(f"initial loss: {loss_history[0]:.6f}")
    print(f"final loss:   {loss_history[-1]:.6f}")
    print()
    print(f"learned bias:    {w[0]:.4f}   (true bias:    {true_bias:.4f})")
    print(f"learned weights: {np.round(w[1:], 4)}")
    print(f"true weights:    {np.round(true_w, 4)}")


if __name__ == "__main__":
    main()
