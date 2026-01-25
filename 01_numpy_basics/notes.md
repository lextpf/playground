# NumPy basics, my notes

First proper go at NumPy. Writing it in my own words so future-me actually
remembers.

## What is an ndarray?

An `ndarray` ("N-dimensional array") is NumPy's main object. It's like a
Python list, but:

- every element has the SAME type
- it has a fixed shape (like 2 rows by 3 columns)
- the math runs way faster because it's done in C under the hood

Attributes I reach for to see what an array looks like:

- `.shape` -> tuple of dimensions, e.g. `(2, 3)`
- `.dtype` -> type of the elements, e.g. `int64` or `float64`
- `.ndim` -> number of dimensions (1 = vector, 2 = matrix)
- `.size` -> total number of elements

## What is a dtype?

The data type shared by every element. I convert with `.astype(...)`. The one
thing to remember: casting a float to an int **chops off** the decimal, it
does NOT round. So `2.9` becomes `2`.

## What is broadcasting?

How NumPy makes arrays of different shapes work together without loops. The
smaller thing gets "stretched" to match the bigger one:

- `array + 5` adds 5 to every element (scalar stretched over the array)
- a `(3, 1)` column plus a `(1, 4)` row makes a `(3, 4)` grid

## What clicked

1. **Vectorization is real.** Squaring a million numbers with a Python loop
   vs `data * data`: the NumPy version was dramatically faster and a single
   line. No more manual loops for math.
2. **Broadcasting saves so much code.** The `(3,1) + (1,4)` example producing
   a full grid is what made it finally make sense.
3. **Seeding matters.** With `np.random.seed(0)` I get the same "random"
   numbers every run, so output is reproducible.

## What still confuses me

Broadcasting with the `(3, 1)` and `(1, 4)` shapes. I kept expecting an error
because the shapes don't match, but NumPy lines them up from the right and
stretches the size-1 dimensions. The rule seems to be "dimensions must be
equal OR one of them must be 1", but I want to test more weird shapes to be
sure I really get it.
