"""Learning goal: get a feel for the four workhorse probability distributions
(uniform, normal, binomial, poisson) by *sampling* from them with a seeded RNG
and confirming that the empirical mean/std of a large sample land close to the
known theoretical (closed-form) values.

Key takeaway I want to lock in: every distribution has parameters that map to
its mean and variance via simple formulas, and a big enough sample recovers
them (the law of large numbers in action).
"""

import numpy as np


def summarize(name: str, samples: np.ndarray,
              theo_mean: float, theo_std: float) -> None:
    """Print empirical vs theoretical mean/std side by side for one sample."""
    emp_mean = float(np.mean(samples))
    emp_std = float(np.std(samples))      # population std (ddof=0)
    print(f"{name}")
    print(f"    mean : empirical={emp_mean:8.4f}   theoretical={theo_mean:8.4f}")
    print(f"    std  : empirical={emp_std:8.4f}   theoretical={theo_std:8.4f}")
    print()


def main() -> None:
    np.random.seed(0)
    n = 100_000   # large sample so empirical estimates are tight

    # --- Uniform(a, b) -------------------------------------------------------
    # mean = (a + b) / 2 ;  variance = (b - a)^2 / 12
    a, b = 2.0, 8.0
    uniform = np.random.uniform(a, b, size=n)
    summarize("Uniform(a=2, b=8)", uniform,
              theo_mean=(a + b) / 2,
              theo_std=np.sqrt((b - a) ** 2 / 12))

    # --- Normal(mu, sigma) ---------------------------------------------------
    # mean = mu ;  std = sigma  (parameters ARE the moments)
    mu, sigma = 5.0, 2.0
    normal = np.random.normal(mu, sigma, size=n)
    summarize("Normal(mu=5, sigma=2)", normal,
              theo_mean=mu, theo_std=sigma)

    # --- Binomial(N, p) ------------------------------------------------------
    # mean = N*p ;  variance = N*p*(1 - p)
    N, p = 20, 0.3
    binomial = np.random.binomial(N, p, size=n)
    summarize("Binomial(N=20, p=0.3)", binomial,
              theo_mean=N * p,
              theo_std=np.sqrt(N * p * (1 - p)))

    # --- Poisson(lambda) -----------------------------------------------------
    # mean = lambda ;  variance = lambda  (mean and variance coincide!)
    lam = 4.0
    poisson = np.random.poisson(lam, size=n)
    summarize("Poisson(lambda=4)", poisson,
              theo_mean=lam, theo_std=np.sqrt(lam))


if __name__ == "__main__":
    main()
