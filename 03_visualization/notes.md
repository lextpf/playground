# Notes: matplotlib basics

Notes from poking at matplotlib. Writing them down so future-me doesn't
re-learn the same lessons.

## Figure vs Axes (the thing that finally clicked)

- A **Figure** is the whole canvas / window, the outer container.
- An **Axes** is a single plot area inside the figure (the box with x/y axes,
  ticks, a title). One figure can hold many Axes (subplots).
- `fig, ax = plt.subplots()` gives me both at once; `plt.subplots(1, 2)` gives
  one figure and an array of two Axes.
- I call methods on the Axes object (`ax.plot`, `ax.set_title`) instead of the
  global `plt.plot`. It's explicit about *which* subplot I'm drawing on. The
  `plt.something` shortcuts secretly act on "the current axes", which gets
  confusing fast once there's more than one subplot.

## Why I save figures instead of showing them in scripts

- `plt.show()` opens an interactive window and **blocks** until I close it.
  Fine in a notebook, but in a plain script it stalls everything (and breaks
  completely on a machine with no display).
- So in scripts I set the backend to **Agg** *before* importing pyplot:

  ```python
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  ```

  Agg renders straight to an image file, so `fig.savefig("out.png")` writes the
  PNG and the script keeps running. Repeatable and headless-friendly. I print
  the saved path at the end so I always know where the file landed.

## Gotchas I actually hit

- **Backend order matters.** `matplotlib.use("Agg")` has to come *before*
  `import matplotlib.pyplot`. If pyplot is already imported, the switch
  silently doesn't take.
- **Overlapping labels.** Titles and axis labels collided between subplots
  until I called `fig.tight_layout()`.
- **`density=True`.** With it on, the histogram bars are normalized so the
  *area* under them is 1, not the raw counts. That's what lets me compare a
  histogram to a probability density. The y-axis is "density", not "count".
- **Forgetting to close figures.** In a loop, every `plt.subplots` leaks
  memory unless I `plt.close(fig)` after saving.
- **Placing text on a plot.** `transform=ax.transAxes` positions a text box in
  axes coordinates (0-1), so it stays pinned to a corner regardless of the data
  range.
- **Seeding.** `np.random.default_rng(0)` so the "random" data is the same
  every run, and the saved plots are reproducible.
