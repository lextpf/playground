# Notes: linear regression from scratch

## The normal equation intuition

I want weights `w` so that `X @ w` lands as close as possible to `y`, where
"close" means smallest sum of squared errors. Take the squared-error cost,
write its derivative, set it to zero, and the algebra collapses to one line:

```
w = (X^T X)^-1 X^T y
```

My mental picture: `y` is a point in space and the columns of `X` span a flat
subspace. The best prediction is the *projection* of `y` onto that subspace,
the closest shadow, and the normal equation is just the formula for that
projection. No learning rate, no loops: it solves for the optimum in one shot.

Two practical things:

- I add a column of ones to `X` so the intercept (bias) is solved as just
  another weight. Cleaner than tracking it separately.
- I use `np.linalg.pinv` instead of `inv`. The pseudo-inverse doesn't blow up
  when `X^T X` is singular or nearly so (e.g. correlated features). The closed
  form is great for small problems, but inverting a big matrix gets expensive,
  which is exactly why gradient descent exists for larger data.

## What R^2 means

R^2 answers: "how much better is my model than just predicting the average?"

```
R^2 = 1 - SS_res / SS_tot
```

- `SS_res` = leftover squared error after my model (what I failed to explain).
- `SS_tot` = squared error of the mean-only baseline.

So R^2 = 1.0 is a perfect fit, R^2 = 0 means no better than guessing the mean,
and negative R^2 means *worse* than the mean (possible on unseen data). It's the
fraction of variance my model explains, and it's scale-free, so unlike raw MSE I
can compare it across datasets.

## Bias vs variance

First time this clicked. Two ways a model can be wrong:

- **High bias (underfitting):** too simple to capture the pattern. A straight
  line through curved data, wrong no matter how much data I give it.
- **High variance (overfitting):** too flexible, chasing the noise. Nails the
  training points but flails on new data.

Plain linear regression sits on the high-bias end: rigid, so it rarely overfits
but it also can't bend to nonlinear truth. I can't minimize both at once,
pushing one down tends to push the other up, and the goal is the sweet spot in
between. Note to self: a perfect training R^2 is *not* automatically good news,
that's where variance hides.
