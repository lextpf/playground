"""Learning goal: practice histograms and reading distributions. I sample
from a normal and an exponential distribution, plot each as a density
histogram, and annotate each subplot with simple stats (mean / std).

Part of my data-viz journal. Focus today: density=True and adding text
to an Axes.
"""

import os

# Backend first, then pyplot (so nothing tries to pop a window open).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sample_data(n=1000):
    """Draw seeded samples from a normal and an exponential distribution."""
    rng = np.random.default_rng(0)

    # Normal: bell-shaped, centered around 0.
    normal_samples = rng.normal(loc=0.0, scale=1.0, size=n)

    # Exponential: skewed, only positive values, long tail to the right.
    exp_samples = rng.exponential(scale=1.0, size=n)

    return normal_samples, exp_samples


def stats_text(samples):
    """Make a tiny stats string I can drop onto a plot."""
    # f-strings keep this readable; round so the box isn't cluttered.
    return f"mean = {samples.mean():.2f}\nstd  = {samples.std():.2f}\nn    = {samples.size}"


def plot_histogram(ax, samples, title, color):
    """Draw one density histogram with a stats text box in the corner."""
    # density=True normalizes the bars so the total area sums to 1.
    # This makes the histogram comparable to a probability density.
    ax.hist(samples, bins=30, density=True, color=color, alpha=0.7,
            edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("density")

    # Put the stats in the upper-right using axes coordinates (0-1 range),
    # so it sticks to the corner no matter the data range.
    ax.text(0.95, 0.95, stats_text(samples),
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))


def build_figure(normal_samples, exp_samples):
    """1x2 figure: normal distribution on the left, exponential on the right."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    plot_histogram(axes[0], normal_samples,
                   "Normal distribution", "tab:blue")
    plot_histogram(axes[1], exp_samples,
                   "Exponential distribution", "tab:green")

    fig.tight_layout()
    return fig


def main():
    normal_samples, exp_samples = sample_data()
    fig = build_figure(normal_samples, exp_samples)

    out_path = os.path.join(os.path.dirname(__file__),
                            "histograms_and_distributions.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
