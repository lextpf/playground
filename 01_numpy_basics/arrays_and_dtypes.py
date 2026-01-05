"""Learning goal: get comfortable making NumPy arrays and understanding
their basic properties (shape, dtype, ndim, size), reshaping, indexing,
slicing, and converting between data types with .astype.

This is day 1 of my NumPy journal, so I'm printing EVERYTHING to see
what each line actually produces.
"""

import numpy as np


def make_arrays():
    """Create arrays a bunch of different ways and print each one."""

    # The most basic way: turn a Python list into a NumPy array.
    a = np.array([1, 2, 3, 4])
    print("np.array from a list:", a)

    # np.arange is like Python's range() but it gives back an array.
    # Start at 0, stop before 10, step by 2.
    b = np.arange(0, 10, 2)
    print("np.arange(0, 10, 2):", b)

    # np.linspace gives evenly spaced numbers between two endpoints.
    # Here: 5 numbers from 0 to 1, and BOTH endpoints are included.
    c = np.linspace(0, 1, 5)
    print("np.linspace(0, 1, 5):", c)

    # np.zeros and np.ones fill an array with 0.0 or 1.0.
    # The (2, 3) means 2 rows and 3 columns.
    zeros = np.zeros((2, 3))
    ones = np.ones((2, 3))
    print("np.zeros((2, 3)):\n", zeros)
    print("np.ones((2, 3)):\n", ones)

    # Random numbers. I seed first so I get the SAME numbers every run.
    # This is important for reproducibility (so my notes match my output).
    np.random.seed(0)
    r = np.random.rand(2, 3)  # uniform random floats in [0, 1)
    print("seeded np.random.rand(2, 3):\n", r)

    return a, zeros


def inspect_array():
    """Look at the attributes that describe an array."""

    # A 2x3 array (2 rows, 3 columns) using arange + reshape.
    m = np.arange(6).reshape(2, 3)
    print("the array m:\n", m)

    # .shape  -> a tuple of (rows, cols)
    print("m.shape:", m.shape)
    # .dtype  -> the data type of the elements (here: integers)
    print("m.dtype:", m.dtype)
    # .ndim   -> number of dimensions (2 because it's a matrix)
    print("m.ndim:", m.ndim)
    # .size   -> total number of elements (2 * 3 = 6)
    print("m.size:", m.size)


def reshape_demo():
    """Take a flat array and change its shape without changing the data."""

    flat = np.arange(12)
    print("flat array:", flat)

    # Reshape into 3 rows and 4 columns. 3 * 4 must equal 12.
    grid = flat.reshape(3, 4)
    print("reshaped to (3, 4):\n", grid)

    # Using -1 lets NumPy figure out that dimension for me.
    # Here I ask for 2 rows and let it compute the columns (12 / 2 = 6).
    auto = flat.reshape(2, -1)
    print("reshaped to (2, -1) -> NumPy fills in 6:\n", auto)


def indexing_and_slicing():
    """Practice grabbing single elements and ranges of elements."""

    m = np.arange(1, 13).reshape(3, 4)
    print("the matrix:\n", m)

    # Single element: [row, col]. Row 0, column 2.
    print("m[0, 2] (row 0, col 2):", m[0, 2])

    # A whole row: row index 1, and ':' means "all columns".
    print("m[1, :] (the whole second row):", m[1, :])

    # A whole column: ':' means all rows, column index 0.
    print("m[:, 0] (the whole first column):", m[:, 0])

    # Slicing a sub-block: rows 0 and 1, columns 1 and 2.
    # Remember the stop index is NOT included.
    print("m[0:2, 1:3] (top middle block):\n", m[0:2, 1:3])


def dtype_casting():
    """Convert an array from one dtype to another with .astype."""

    floats = np.array([1.9, 2.5, 3.1])
    print("original float array:", floats, "dtype:", floats.dtype)

    # Casting to int just CHOPS OFF the decimal part (it does not round).
    # So 1.9 becomes 1 and 2.5 becomes 2. Good to remember!
    as_int = floats.astype(np.int64)
    print("after .astype(np.int64):", as_int, "dtype:", as_int.dtype)

    # Going the other way: ints turned into floats.
    ints = np.array([1, 2, 3])
    as_float = ints.astype(np.float64)
    print("ints cast to float:", as_float, "dtype:", as_float.dtype)


def main():
    print("===== making arrays =====")
    make_arrays()
    print("\n===== inspecting an array =====")
    inspect_array()
    print("\n===== reshaping =====")
    reshape_demo()
    print("\n===== indexing and slicing =====")
    indexing_and_slicing()
    print("\n===== dtype casting =====")
    dtype_casting()


if __name__ == "__main__":
    main()
