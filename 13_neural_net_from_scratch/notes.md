# Neural nets from scratch, my notes

## What I built today
- `activations.py`: sigmoid / relu / tanh and their derivatives, plus a
  finite-difference check so I actually trust the analytic gradients.
- `perceptron.py`: a single linear unit that learns AND.
- `mlp.py`: one hidden layer, forward + backprop + gradient descent.
- `xor_experiment.py`: the payoff, the MLP learns XOR.

## Why XOR needs a hidden layer
A single perceptron computes `step(w . x + b)`. That's *one* straight line
splitting the plane into two halves. For AND, the single positive case (1,1)
sits in its own corner, so one line works fine. XOR is different: the
positives are (0,1) and (1,0), the negatives are (0,0) and (1,1). The two
classes sit on opposite diagonals, no single straight line can separate
them. This is the classic Minsky & Papert result that stalled the field.

A hidden layer fixes this because each hidden unit draws its own line, and the
output layer combines those lines. With two hidden boundaries I can carve out
the diagonal "exactly one is on" region. The hidden nonlinearity (tanh here) is
essential: stacking purely *linear* layers just collapses back into one linear
map, so I'd still only get a straight boundary.

## Deriving backprop (the intuition that stuck)
Backprop is just the chain rule applied layer by layer, reusing work.
Forward: `z1 = X W1 + b1`, `a1 = tanh(z1)`, `z2 = a1 W2 + b2`, `a2 = sigmoid(z2)`.

Going backward, I want `dL/dW` for each weight. The trick is to carry an
"error signal" `dz` for each layer:
- Output: with sigmoid output + binary cross-entropy, the gradient of the loss
  w.r.t. the pre-activation collapses to the beautifully simple `dz2 = a2 - y`.
  (The sigmoid derivative and the BCE derivative cancel, this is *why* that
  pairing is the default.)
- A layer's weight gradient is always `(input to that layer).T @ dz`.
- To push error to the previous layer: `dz_prev = (dz @ W.T) * f'(z_prev)`.
  Multiply by the local activation derivative to "undo" the nonlinearity.

## What finally made it click
For a long time backprop felt like opaque matrix bookkeeping. Two things flipped
it for me:

1. **`dz` is reusable error, not magic.** Each layer receives an error signal
   from above, scales its own weight gradient with it, then hands a transformed
   error signal downward. It's a conveyor belt, not a monolith.
2. **The finite-difference check.** Comparing my analytic `sigmoid_derivative`
   to `(f(x+h) - f(x-h)) / 2h` and watching the error come out ~1e-8 proved the
   gradients were *correct*, not just plausible. Once I trusted the building
   block, trusting the composed backward pass was easy.

Next: swap in ReLU hidden units, add mini-batching, and try a 2-hidden-layer
net on a small synthetic blobs dataset.
