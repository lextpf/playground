"""Learning goal: see WHY feature scaling matters for gradient descent.

The bug: if features live on wildly different scales, the MSE loss surface is
stretched into a long narrow valley. A single learning rate can't suit every
direction at once, so the steps overshoot along the steep axis and the loss
explodes to inf/nan -> divergence.

The fix: standardize each feature to zero mean and unit variance (z-score).
Now every direction has a comparable scale, the valley is roughly round, and
the SAME learning rate converges nicely.

I reuse the core gradient descent from gradient_descent.py to keep the lesson
focused on the scaling, not on re-deriving the math.
"""

import numpy as np

from gradient_descent import gradient_descent, add_bias_column, mse_loss


def make_badly_scaled_data(n_samples=200, seed=0):
    """Three features deliberately on very different scales.

    Feature 0 ~ thousands, feature 1 ~ ones, feature 2 ~ thousandths.
    This kind of mismatch is exactly what breaks naive gradient descent.
    """
    rng = np.random.default_rng(seed)
    f0 = rng.normal(loc=5000.0, scale=1500.0, size=n_samples)
    f1 = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    f2 = rng.normal(loc=0.0, scale=0.001, size=n_samples)
    X = np.column_stack([f0, f1, f2])

    true_w = np.array([0.01, 3.0, 500.0])
    y = X @ true_w + 10.0 + 0.5 * rng.normal(size=n_samples)
    return X, y


def standardize(X):
    """Z-score each column: (x - mean) / std.

    I guard the std against zero to avoid divide-by-zero on a constant
    feature, a small numerical-stability habit I'm trying to build.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return (X - mean) / std_safe


def final_loss(X, y, lr, n_iters=500):
    """Run gradient descent and report the final loss.

    Loss can become inf/nan when training diverges; we return that as-is so
    the caller can detect divergence rather than crashing.
    """
    w, history = gradient_descent(X, y, lr=lr, n_iters=n_iters)
    return history[-1], w


def main():
    np.random.seed(0)
    X_raw, y = make_badly_scaled_data(seed=0)

    lr = 0.01  # one learning rate used for BOTH runs, to isolate scaling

    # --- BUG: train on raw, badly-scaled features ---
    diverged_loss, _ = final_loss(X_raw, y, lr=lr)

    # --- FIX: standardize the features, same learning rate ---
    X_std = standardize(X_raw)
    converged_loss, _ = final_loss(X_std, y, lr=lr)

    print("Feature scaling and gradient descent")
    print(f"learning rate (same for both): {lr}")
    print()
    print(f"BEFORE (raw features) final loss: {diverged_loss}")
    diverged = not np.isfinite(diverged_loss)
    print(f"  -> diverged? {diverged}")
    print()
    print(f"AFTER  (z-scored)     final loss: {converged_loss:.6f}")
    print(f"  -> converged? {np.isfinite(converged_loss)}")
    print()
    print("Lesson: same lr, same data, same code, only scaling changed.")


if __name__ == "__main__":
    main()
