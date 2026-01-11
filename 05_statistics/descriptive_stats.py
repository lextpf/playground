"""Learning goal: implement the core descriptive statistics (mean, variance,
standard deviation, median, percentiles) FROM SCRATCH using only low-level
numpy primitives (sum, sort, indexing), then cross-check every result against
the trusted numpy built-in with np.allclose.

Why bother re-deriving these? Re-implementing the formulas is the fastest way
to internalize the difference between *population* and *sample* variance (the
n vs. n-1 denominator) and how percentiles are actually interpolated.
"""

import numpy as np


def my_mean(x: np.ndarray) -> float:
    """Arithmetic mean: sum of values divided by the count."""
    return float(np.sum(x) / x.size)


def my_variance(x: np.ndarray, ddof: int = 0) -> float:
    """Variance = average squared deviation from the mean.

    ddof ("delta degrees of freedom") picks the denominator:
      ddof=0 -> divide by n      (population variance)
      ddof=1 -> divide by n - 1  (sample variance, Bessel's correction)
    """
    mu = my_mean(x)
    squared_deviations = (x - mu) ** 2
    return float(np.sum(squared_deviations) / (x.size - ddof))


def my_std(x: np.ndarray, ddof: int = 0) -> float:
    """Standard deviation is just the square root of the variance."""
    return float(np.sqrt(my_variance(x, ddof=ddof)))


def my_median(x: np.ndarray) -> float:
    """Median = middle value of the sorted data.

    Even-length arrays have no single middle, so we average the two
    central elements.
    """
    s = np.sort(x)
    n = s.size
    mid = n // 2
    if n % 2 == 1:           # odd length -> exact middle element
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2)  # even -> mean of the two middles


def my_percentile(x: np.ndarray, q: float) -> float:
    """The q-th percentile (0 <= q <= 100) via linear interpolation.

    This mirrors numpy's default ('linear') method: map q onto a fractional
    index over the sorted data, then linearly blend the neighbouring values.
    """
    s = np.sort(x)
    rank = (q / 100.0) * (s.size - 1)   # fractional position in [0, n-1]
    lo = int(np.floor(rank))
    hi = int(np.ceil(rank))
    frac = rank - lo                    # how far between lo and hi we sit
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def check(name: str, mine: float, reference: float) -> None:
    """Compare one of my values against numpy and print a PASS/FAIL line."""
    ok = np.allclose(mine, reference)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name:<22} mine={mine:.6f}  numpy={reference:.6f}")


def main() -> None:
    # Reproducible synthetic data so every run prints identical numbers.
    np.random.seed(0)
    data = np.random.normal(loc=5.0, scale=2.0, size=50)

    check("mean", my_mean(data), np.mean(data))
    check("variance (pop)", my_variance(data, ddof=0), np.var(data, ddof=0))
    check("variance (sample)", my_variance(data, ddof=1), np.var(data, ddof=1))
    check("std (pop)", my_std(data, ddof=0), np.std(data, ddof=0))
    check("std (sample)", my_std(data, ddof=1), np.std(data, ddof=1))
    check("median", my_median(data), np.median(data))
    for q in (25, 50, 75, 90):
        check(f"percentile p{q}", my_percentile(data, q), np.percentile(data, q))


if __name__ == "__main__":
    main()
