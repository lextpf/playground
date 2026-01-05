# NumPy Day 1 - Learning Notes

These are my notes from my first real session with NumPy. Writing them
in my own words so future-me actually remembers.

## What is an ndarray?

An `ndarray` ("N-dimensional array") is NumPy's main object. It's like a
Python list, but:

- every element has the SAME type
- it has a fixed shape (like 2 rows by 3 columns)
- the math runs way faster because it's done in C under the hood

I can check what an array looks like with these attributes:

- `.shape` -> tuple of dimensions, e.g. `(2, 3)`
- `.dtype` -> the type of the elements, e.g. `int64` or `float64`
- `.ndim` -> how many dimensions (1 = vector, 2 = matrix)
- `.size` -> total number of elements

## What is a dtype?

A `dtype` is the data type of every element in the array (since they all
share one). Common ones I saw: `int64`, `float64`. I can convert with
`.astype(...)`. The thing I have to remember: casting a float to an int
**chops off** the decimal, it does NOT round. So `2.9` becomes `2`.

## What is broadcasting?

Broadcasting is how NumPy makes arrays of different shapes work together
without me writing loops. The smaller thing gets "stretched" to match the
bigger one:

- `array + 5` adds 5 to every element (scalar stretched over the array)
- a `(3, 1)` column plus a `(1, 4)` row makes a `(3, 4)` grid

## Biggest things that clicked today

1. **Vectorization is real.** I squared a million numbers with a Python
   loop and then with `data * data`. The NumPy version was dramatically
   faster, and the code was a single line. No more manual loops for math.
2. **Broadcasting saves so much code.** The `(3,1) + (1,4)` example
   producing a full grid made it finally make sense.
3. **Seeding matters.** With `np.random.seed(0)` I get the same "random"
   numbers every run, so my output is reproducible.

## One thing that confused me

Broadcasting rules with the `(3, 1)` and `(1, 4)` shapes. I kept expecting
an error because the shapes don't match, but NumPy lined them up from the
right and stretched the size-1 dimensions. I think the rule is "dimensions
must be equal OR one of them must be 1" but I want to test more weird
shapes tomorrow to be sure I really get it.
