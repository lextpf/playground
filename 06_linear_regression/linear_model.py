"""Learning goal: take yesterday's explicit script and refactor the regression
into a small, reusable LinearRegression class with fit/predict/score.

First real refactor for me, the win is that the normal-equation math gets
hidden behind a tidy interface that looks a bit like scikit-learn, so I can
reuse it without copy-pasting the linear algebra every time.
"""

import numpy as np


class LinearRegression:
    """Ordinary least squares linear regression via the normal equation.

    The model learns weights w and an intercept b such that
    y_hat = X @ w + b. Internally it solves the closed-form solution
    w = (X^T X)^-1 X^T y on an augmented matrix that includes a bias column,
    so there's no iterative training loop.

    Attributes:
        coef_:      learned feature weights, shape (n_features,).
        intercept_: learned bias term (scalar).
    """

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    @staticmethod
    def _add_bias(X):
        """Prepend a ones column so the intercept is solved alongside weights."""
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def fit(self, X, y):
        """Fit the model to data using the normal equation.

        Args:
            X: array of shape (n_samples, n_features).
            y: array of shape (n_samples,) or (n_samples, 1).

        Returns:
            self, so calls can be chained.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        X_b = self._add_bias(X)
        w = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.intercept_ = w[0]
        self.coef_ = w[1:]
        return self

    def predict(self, X):
        """Predict targets for X using the learned weights and intercept."""
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """Return the R^2 of predictions on (X, y), computed from scratch."""
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot


def make_data(n_samples=120, n_features=3, noise=0.8):
    """Synthetic multi-feature linear data with known weights and bias."""
    X = np.random.uniform(-3, 3, size=(n_samples, n_features))
    true_w = np.array([2.0, -1.5, 0.5])
    true_b = 4.0
    y = X @ true_w + true_b + np.random.normal(0, noise, size=n_samples)
    return X, y, true_w, true_b


def main():
    np.random.seed(0)

    X, y, true_w, true_b = make_data()
    model = LinearRegression().fit(X, y)

    print("True weights :", np.round(true_w, 3))
    print("Learned coef_:", np.round(model.coef_, 3))
    print(f"True bias : {true_b:.3f}")
    print(f"Learned intercept_: {model.intercept_:.3f}")
    print(f"R^2 score: {model.score(X, y):.4f}")


if __name__ == "__main__":
    main()
