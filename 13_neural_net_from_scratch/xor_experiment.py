"""Learning XOR with the MLP, the problem a single perceptron cannot solve.

Learning goal: demonstrate empirically that adding a hidden layer lets the
network represent a non-linearly-separable function. XOR's two positive cases
sit on opposite corners of the square, so no single straight line separates
them, but a hidden layer composes two boundaries into a nonlinear region.
"""

from __future__ import annotations

import numpy as np

from mlp import MLP


def main() -> None:
    np.random.seed(0)
    # XOR truth table: output is 1 iff exactly one input is 1.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=int)

    model = MLP(n_inputs=2, n_hidden=4, lr=0.5, seed=0)
    history = model.fit(X, y, epochs=5000)

    probs = model.forward(X)["a2"].ravel()
    preds = model.predict(X).ravel()
    accuracy = float(np.mean(preds == y))

    print("MLP learning XOR (not linearly separable)")
    print(f"  start loss={history[0]:.4f}  final loss={history[-1]:.4f}")
    for xi, target, prob, pred in zip(X, y, probs, preds):
        print(f"  input={xi.astype(int)} -> prob={prob:.3f} pred={pred} (target={target})")
    print(f"  accuracy={accuracy:.0%}")

    assert accuracy == 1.0, "MLP failed to learn XOR"
    print("  The hidden layer solved XOR, exactly what the perceptron could not.")


if __name__ == "__main__":
    main()
