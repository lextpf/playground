"""Reusable activation functions and their derivatives.

Learning goal: build the small library of nonlinearities a neural net needs,
each paired with its analytic derivative, and sanity-check one derivative
numerically with finite differences so I trust the math before wiring up
backprop in mlp.py.

Convention: every derivative takes the *pre-activation* input ``x`` (not the
already-activated output), so it can be composed cleanly in a backward pass.
"""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid, squashing inputs to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid w.r.t. its input: s(x) * (1 - s(x))."""
    s = sigmoid(x)
    return s * (1.0 - s)


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit: max(0, x)."""
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 where x > 0, else 0 (0 at the kink by convention)."""
    return (x > 0.0).astype(x.dtype)


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent, squashing inputs to (-1, 1)."""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh: 1 - tanh(x)^2."""
    return 1.0 - np.tanh(x) ** 2


def _finite_difference_check() -> None:
    """Numerically verify sigmoid_derivative via central differences.

    The central-difference approximation f'(x) ~= (f(x+h) - f(x-h)) / (2h)
    is O(h^2) accurate, so it should match the analytic derivative closely.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=10)
    h = 1e-6
    numerical = (sigmoid(x + h) - sigmoid(x - h)) / (2.0 * h)
    analytic = sigmoid_derivative(x)
    max_err = float(np.max(np.abs(numerical - analytic)))
    assert max_err < 1e-7, f"derivative mismatch: max error {max_err}"
    print(f"sigmoid_derivative finite-difference check passed (max err {max_err:.2e})")


if __name__ == "__main__":
    _finite_difference_check()
