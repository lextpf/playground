"""Learning goal: see the Central Limit Theorem (CLT) with my own eyes,
numerically.

The CLT says: if you take the *mean* of n i.i.d. samples, the distribution of
those sample means tends toward a normal distribution as n grows, no matter
how non-normal the original distribution is. To make the effect obvious I draw
from the exponential distribution, which is strongly right-skewed.

I track skewness as the headline number: the exponential has skewness ~2, but
as the sample size n increases the skewness of the sample means should collapse
toward 0 (the signature of a symmetric, normal-shaped distribution).
"""

import os

import numpy as np


def skewness(x: np.ndarray) -> float:
    """Fisher-Pearson sample skewness: the 3rd standardized moment.

    skew = mean( ((x - mu) / sigma)^3 ). Zero means symmetric; positive means
    a longer right tail (which is what the raw exponential has).
    """
    mu = np.mean(x)
    sigma = np.std(x)           # population std
    return float(np.mean(((x - mu) / sigma) ** 3))


def sample_means(rng: np.random.Generator, scale: float,
                 sample_size: int, num_experiments: int) -> np.ndarray:
    """Run `num_experiments` trials; each averages `sample_size` exponential
    draws. Returns the array of resulting sample means."""
    # Shape (num_experiments, sample_size): one row per experiment.
    draws = rng.exponential(scale=scale, size=(num_experiments, sample_size))
    return draws.mean(axis=1)   # collapse each row to its mean


def main() -> None:
    np.random.seed(0)                    # seed global state for good measure
    rng = np.random.default_rng(0)       # explicit generator, also seeded

    scale = 1.0                          # exponential: mean = std = scale
    num_experiments = 20_000

    # Baseline: skewness of the RAW (un-averaged) exponential distribution.
    raw = rng.exponential(scale=scale, size=num_experiments)
    print(f"Raw exponential (scale={scale}):")
    print(f"    mean={np.mean(raw):.4f}  std={np.std(raw):.4f}  "
          f"skew={skewness(raw):.4f}  (theory: skew=2.0)\n")

    print("Distribution of SAMPLE MEANS as sample size n grows:")
    print(f"{'n':>6} | {'mean':>8} | {'std':>8} | {'skew':>8}")
    print("-" * 40)

    skews = {}
    for n in (1, 2, 5, 30, 100, 1000):
        means = sample_means(rng, scale, sample_size=n,
                             num_experiments=num_experiments)
        # CLT prediction: std of the means ~ scale / sqrt(n).
        skews[n] = skewness(means)
        print(f"{n:>6} | {np.mean(means):8.4f} | "
              f"{np.std(means):8.4f} | {skews[n]:8.4f}")

    print("-" * 40)
    print(f"Skew at n=1   : {skews[1]:.4f}  (still very skewed, ~ raw)")
    print(f"Skew at n=1000: {skews[1000]:.4f}  (near 0 -> looks normal!)")
    assert abs(skews[1000]) < abs(skews[1]), "skew should shrink as n grows"
    print("\nConfirmed: averaging washes out skew -> CLT in action.")

    # Optional visual: histogram of sample means for the largest n.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        means_big = sample_means(rng, scale, sample_size=1000,
                                 num_experiments=num_experiments)
        fig, ax = plt.subplots()
        ax.hist(means_big, bins=60, color="steelblue", edgecolor="white")
        ax.set_title("Distribution of sample means (n=1000) -> bell shaped")
        ax.set_xlabel("sample mean")
        ax.set_ylabel("frequency")
        out_path = os.path.join(os.path.dirname(__file__), "central_limit_demo.png")
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {out_path}")
    except ImportError:
        print("matplotlib not available; skipped the histogram plot.")


if __name__ == "__main__":
    main()
