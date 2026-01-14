"""Learning goal: a tiny, dependency-free sanity harness for today's code.

No pytest, just plain `assert`s so I can run `python test_logistic_regression.py`
and immediately see whether the building blocks and the trained model behave.
Each check targets one property I claimed in the other modules.
"""

from __future__ import annotations

import numpy as np

from logistic_regression import LogisticRegression, accuracy, make_dataset
from sigmoid_and_loss import sigmoid


def check_sigmoid_midpoint() -> None:
    # sigmoid(0) is exactly 0.5 by definition.
    assert abs(float(sigmoid(np.array([0.0]))[0]) - 0.5) < 1e-12


def check_sigmoid_monotonic() -> None:
    zs = np.linspace(-10, 10, 50)
    s = sigmoid(zs)
    # Strictly increasing -> every consecutive diff is positive.
    assert np.all(np.diff(s) > 0), "sigmoid should be monotonically increasing"
    # And it stays inside the open interval (0, 1).
    assert np.all(s > 0) and np.all(s < 1)


def check_model_learns() -> None:
    np.random.seed(0)
    X, y = make_dataset(n=200, seed=0)
    model = LogisticRegression(lr=0.1, n_iters=500).fit(X, y)

    preds = model.predict(X)
    # Predictions must be hard labels in {0, 1}.
    assert set(np.unique(preds)).issubset({0, 1}), "predictions must be 0/1"

    acc = accuracy(y, preds)
    assert acc > 0.8, f"expected >0.8 accuracy on easy data, got {acc:.3f}"


def main() -> None:
    check_sigmoid_midpoint()
    check_sigmoid_monotonic()
    check_model_learns()
    print("all checks passed")


if __name__ == "__main__":
    main()
