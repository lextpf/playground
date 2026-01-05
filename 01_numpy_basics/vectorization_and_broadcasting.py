"""Learning goal: understand WHY NumPy is fast (vectorization) and how
broadcasting lets arrays of different shapes work together.

First I do the same math two ways (a slow Python loop vs. a NumPy
vectorized version) and time them. Then I play with broadcasting.
"""

import time

import numpy as np


def python_loop_square(values):
    """Square every number using a plain Python for-loop (the slow way)."""
    result = []
    for v in values:
        # Do the operation one element at a time in pure Python.
        result.append(v * v)
    return result


def compare_speed():
    """Time the loop version against the vectorized version."""

    # Seed so the random data is identical every run.
    np.random.seed(0)

    n = 1_000_000  # one million numbers, big enough to feel the difference
    data = np.random.rand(n)

    # --- Slow way: Python for-loop ---
    start = time.perf_counter()
    loop_result = python_loop_square(data)
    loop_time = time.perf_counter() - start
    print(f"Python loop took:      {loop_time:.4f} seconds")

    # --- Fast way: NumPy vectorized (no explicit loop!) ---
    # NumPy does the loop in fast compiled C code under the hood.
    start = time.perf_counter()
    vector_result = data * data
    vector_time = time.perf_counter() - start
    print(f"NumPy vectorized took: {vector_time:.4f} seconds")

    # Sanity check: both methods should give the same answer.
    # np.allclose checks the numbers are equal (allowing tiny float error).
    print("results match:", np.allclose(loop_result, vector_result))

    # How many times faster was NumPy? Guard against divide-by-zero just in case.
    if vector_time > 0:
        speedup = loop_time / vector_time
        print(f"NumPy was about {speedup:.1f}x faster!")


def scalar_broadcasting():
    """Broadcasting case 1: a single number combined with a whole array."""

    arr = np.array([10, 20, 30])
    print("array:", arr, "shape:", arr.shape)

    # Adding a scalar adds it to EVERY element. The scalar is "stretched"
    # to match the array's shape. This is the simplest broadcasting.
    print("arr + 5:", arr + 5)
    print("arr * 2:", arr * 2)


def shape_broadcasting():
    """Broadcasting case 2: combining a column and a row to make a grid."""

    # A column vector: shape (3, 1) -> 3 rows, 1 column.
    col = np.array([[0], [10], [20]])
    print("col shape:", col.shape)
    print(col)

    # A row vector: shape (1, 4) -> 1 row, 4 columns.
    row = np.array([[1, 2, 3, 4]])
    print("row shape:", row.shape)
    print(row)

    # When we add them, NumPy stretches the column across the columns and
    # the row across the rows, giving a (3, 4) result. Mind = blown.
    result = col + row
    print("col + row shape:", result.shape)
    print("col + row result:\n", result)


def main():
    print("===== vectorization speed test =====")
    compare_speed()
    print("\n===== scalar broadcasting =====")
    scalar_broadcasting()
    print("\n===== (3,1) + (1,4) broadcasting =====")
    shape_broadcasting()


if __name__ == "__main__":
    main()
