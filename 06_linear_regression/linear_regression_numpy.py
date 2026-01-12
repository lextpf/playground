"""Learning goal: fit linear regression from scratch using the closed-form
normal equation, then judge the fit with an R^2 score I compute by hand.

This is the "do it the explicit way once" version. No class yet, just
functions so I can see every step: build data, solve normal equation, score.
"""

import numpy as np


def make_data(n_samples=100, true_slope=3.5, true_bias=-1.2, noise=1.0):
    """Make noisy data from a known line y = slope*x + bias + noise.

    Returns X (column vector of features) and y (target vector). Knowing the
    true parameters lets me check whether the fit recovers them.
    """
    x = np.random.uniform(-5, 5, size=(n_samples, 1))
    noise_vec = np.random.normal(0, noise, size=(n_samples, 1))
    y = true_slope * x + true_bias + noise_vec
    return x, y


def add_bias_column(X):
    """Prepend a column of ones so the intercept becomes just another weight."""
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def normal_equation(X_b, y):
    """Closed-form least squares: w = (X^T X)^-1 X^T y.

    X_b already has the bias column, so w[0] is the intercept and the rest are
    the feature coefficients. I use pinv instead of inv to stay safe if X^T X
    is (nearly) singular.
    """
    return np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y


def r_squared(y_true, y_pred):
    """R^2 = 1 - SS_res / SS_tot, computed from scratch.

    SS_res is how much variance my model leaves unexplained; SS_tot is the
    variance around the mean (the dumb baseline). 1.0 is perfect.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def main():
    np.random.seed(0)

    true_slope, true_bias = 3.5, -1.2
    X, y = make_data(true_slope=true_slope, true_bias=true_bias)

    X_b = add_bias_column(X)
    w = normal_equation(X_b, y)
    learned_bias, learned_slope = w[0, 0], w[1, 0]

    y_pred = X_b @ w
    score = r_squared(y, y_pred)

    print("Learned vs true parameters:")
    print(f"  bias : learned={learned_bias:+.4f}  true={true_bias:+.4f}")
    print(f"  slope: learned={learned_slope:+.4f}  true={true_slope:+.4f}")
    print(f"R^2 on training data: {score:.4f}")


if __name__ == "__main__":
    main()
