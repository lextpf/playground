"""Learning goal: get comfortable with matplotlib subplots by drawing a
line plot (noisy sine wave) and a scatter plot (two correlated variables)
side by side, then saving the figure to a PNG instead of showing it.

This is part of my data-viz learning journal. I'm practicing wrapping
plotting logic into small functions so each piece does one thing.
"""

import os

# Always set the backend BEFORE importing pyplot.
# "Agg" is a non-interactive backend that writes to files -> good for scripts.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_data(n=200):
    """Build the synthetic data I'll plot.

    Returns the x values, a noisy sine wave (for the line plot), and a
    pair of correlated variables (for the scatter plot).
    """
    # Seed so I get the exact same "random" data every run. Reproducibility!
    rng = np.random.default_rng(0)

    # Line data: evenly spaced x, sine of x, plus a little noise on top.
    x = np.linspace(0, 4 * np.pi, n)
    noisy_sine = np.sin(x) + rng.normal(0, 0.2, size=n)

    # Scatter data: y is roughly proportional to x, again with noise.
    # This makes the two variables visibly correlated.
    scatter_x = rng.normal(0, 1, size=n)
    scatter_y = 2.0 * scatter_x + rng.normal(0, 1, size=n)

    return x, noisy_sine, scatter_x, scatter_y


def build_figure(x, noisy_sine, scatter_x, scatter_y):
    """Create a 1x2 figure: line plot on the left, scatter on the right."""
    # plt.subplots gives me one Figure and an array of Axes (one per subplot).
    # The Figure is the whole canvas; each Axes is a single plot area.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # --- Left subplot: the noisy sine wave as a line ---
    ax_line = axes[0]
    ax_line.plot(x, noisy_sine, color="tab:blue", linewidth=1)
    ax_line.set_title("Noisy sine wave")
    ax_line.set_xlabel("x")
    ax_line.set_ylabel("sin(x) + noise")
    ax_line.grid(True, alpha=0.3)

    # --- Right subplot: the correlated variables as a scatter ---
    ax_scatter = axes[1]
    ax_scatter.scatter(scatter_x, scatter_y, color="tab:orange", s=15, alpha=0.6)
    ax_scatter.set_title("Two correlated variables")
    ax_scatter.set_xlabel("x")
    ax_scatter.set_ylabel("y ~ 2x + noise")
    ax_scatter.grid(True, alpha=0.3)

    # tight_layout keeps the titles/labels from overlapping.
    fig.tight_layout()
    return fig


def main():
    x, noisy_sine, scatter_x, scatter_y = make_data()
    fig = build_figure(x, noisy_sine, scatter_x, scatter_y)

    # Save into THIS script's folder so the PNG lands next to the code.
    out_path = os.path.join(os.path.dirname(__file__), "line_and_scatter.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)  # free the figure's memory once it's written

    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
