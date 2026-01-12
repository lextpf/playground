# Notes - Linear Regression from Scratch

## The normal equation intuition

I'm trying to find weights `w` so that `X @ w` lands as close as possible to
`y`. "Close" means smallest sum of squared errors. If I take the squared-error
cost, write down its derivative, and set it to zero, the algebra collapses to
one clean line:

```
w = (X^TX)^-1 X^T y
```

My mental picture: `y` is a point in space, and the columns of `X` span a
flat subspace. The best prediction is the *projection* of `y` onto that
subspace, the closest shadow. The normal equation is just the formula for
that projection. No learning rate, no loops, no iterations: it solves for the
optimum in one shot.

Two practical things I learned:

- I add a column of ones to `X` so the intercept (bias) is solved as just
  another weight. Cleaner than tracking it separately.
- I use `np.linalg.pinv` instead of `inv`. The pseudo-inverse doesn't blow up
  when `X^TX` is singular or nearly so (e.g. correlated features). The closed
  form is great for small problems but inverting a big matrix gets expensive,
  which is exactly why gradient descent exists for larger data.

## What R^2 means

R^2 answers: "how much better is my model than just predicting the average?"

```
R^2 = 1 - SS_res / SS_tot
```

- `SS_res` = leftover squared error after my model (what I failed to explain).
- `SS_tot` = squared error of the dumb mean-only baseline.

So R^2 = 1.0 is a perfect fit, R^2 = 0 means I'm no better than guessing the
mean, and negative R^2 means I'm actually *worse* than the mean (possible on
unseen data). It's the fraction of variance my model explains. Handy because
it's scale-free, unlike raw MSE, I can compare it across datasets.

## Early note: bias vs variance

First time this clicked a little. Two ways a model can be wrong:

- **High bias (underfitting):** model too simple to capture the pattern. A
  straight line through curved data, wrong no matter how much data I give it.
- **High variance (overfitting):** model too flexible, chasing the noise. It
  nails the training points but flails on new data.

Plain linear regression sits on the high-bias end: very rigid, so it rarely
overfits but it also can't bend to nonlinear truth. The tradeoff is that I
can't minimize both at once, pushing one down tends to push the other up,
and the goal is the sweet spot in the middle. Note to self: a perfect training
R^2 is *not* automatically good news; that's where variance hides.
