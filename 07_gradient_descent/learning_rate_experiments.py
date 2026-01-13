"""Learning goal: build intuition for how the learning rate controls
gradient descent.

I sweep several learning rates on the SAME seeded data and print a small table
of lr -> final_loss / converged?. I expect to see three regimes:
  - too small  -> converges, but barely moves in the iteration budget (slow),
  - good       -> low final loss (genuinely converged),
  - too large  -> overshoots and the loss explodes (diverges).

This makes the classic "Goldilocks" picture of the learning rate concrete.
"""

import numpy as np

from gradient_descent import gradient_descent, make_synthetic_data, mse_loss, add_bias_column


def classify_run(initial_loss, final_loss, tol=1e-3):
    """Label a run as diverged / good / slow based on its loss behaviour.

    - diverged: final loss is not finite (inf/nan) -> blew up,
    - good:     converged to a small loss (well below the starting loss),
    - slow:     still finite but hasn't made much progress yet.
    """
    if not np.isfinite(final_loss):
        return "DIVERGED (too large)"
    if final_loss < tol:
        return "good (converged)"
    if final_loss > 0.5 * initial_loss:
        return "slow (too small)"
    return "ok (converging)"


def run_sweep(X, y, learning_rates, n_iters=500):
    """Run gradient descent for each learning rate; collect results.

    Returns a list of dict rows so printing the table stays separate from
    the computation.
    """
    rows = []
    for lr in learning_rates:
        _, history = gradient_descent(X, y, lr=lr, n_iters=n_iters)
        initial_loss = history[0]
        final_loss = history[-1]
        rows.append({
            "lr": lr,
            "final_loss": final_loss,
            "verdict": classify_run(initial_loss, final_loss),
        })
    return rows


def print_table(rows):
    """Pretty-print the sweep as an aligned table."""
    print(f"{'learning_rate':>14} | {'final_loss':>16} | verdict")
    print("-" * 56)
    for r in rows:
        # inf/nan won't format with a fixed-point spec, so handle separately
        if np.isfinite(r["final_loss"]):
            loss_str = f"{r['final_loss']:16.6f}"
        else:
            loss_str = f"{str(r['final_loss']):>16}"
        print(f"{r['lr']:>14g} | {loss_str} | {r['verdict']}")


def main():
    np.random.seed(0)
    X, y, _, _ = make_synthetic_data(seed=0)

    # spanning many orders of magnitude to hit all three regimes
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 1.1]

    rows = run_sweep(X, y, learning_rates, n_iters=500)

    print("Learning rate sweep (batch GD, linear regression, 500 iters)")
    print()
    print_table(rows)
    print()
    print("Takeaway: there is a sweet spot, too small wastes iterations,")
    print("too large overshoots and diverges.")


if __name__ == "__main__":
    main()
