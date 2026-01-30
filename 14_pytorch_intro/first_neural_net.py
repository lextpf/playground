"""Learning goal: my first torch.nn classifier with a clean training loop.

Plan: make a seeded, linearly-not-quite-separable 2-class dataset with numpy,
wrap a tiny MLP in an nn.Module, then run the canonical PyTorch loop
(zero_grad -> forward -> loss -> backward -> step) and watch the loss fall.
I print loss every so often and the final training accuracy. CPU-only.
"""

import random

import numpy as np
import torch
import torch.nn as nn


def make_dataset(n_samples: int = 400, seed: int = 0):
    """Two Gaussian blobs (one per class) that overlap a bit, via numpy RNG.

    Returns float32 features and int64 labels as tensors, the dtypes that
    nn.Linear and nn.CrossEntropyLoss expect respectively.
    """
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    # Class 0 centered at (-1.5, -1.5), class 1 at (+1.5, +1.5); shared spread.
    c0 = rng.normal(loc=-1.5, scale=1.3, size=(half, 2))
    c1 = rng.normal(loc=1.5, scale=1.3, size=(half, 2))

    X = np.vstack([c0, c1]).astype(np.float32)
    y = np.concatenate([np.zeros(half), np.ones(half)]).astype(np.int64)

    # Shuffle so the two classes aren't blocked together.
    perm = rng.permutation(n_samples)
    X, y = X[perm], y[perm]
    return torch.from_numpy(X), torch.from_numpy(y)


class MLP(nn.Module):
    """A small feedforward net: 2 -> 16 -> 16 -> 2 logits."""

    def __init__(self, in_dim: int = 2, hidden: int = 16, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return raw logits; CrossEntropyLoss applies log-softmax internally.
        return self.net(x)


@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction correct. no_grad since we don't need the graph for eval."""
    preds = model(X).argmax(dim=1)
    return (preds == y).float().mean().item()


def train(model, X, y, epochs: int = 200, lr: float = 0.05) -> None:
    """The idiomatic PyTorch training loop (full-batch here for simplicity)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()       # clear grads from the previous step
        logits = model(X)           # forward pass
        loss = criterion(logits, y) # compare logits to integer targets
        loss.backward()             # backprop
        optimizer.step()            # update weights

        if epoch == 1 or epoch % 40 == 0:
            print(f"epoch {epoch:3d} | loss {loss.item():.4f}")


def main() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    X, y = make_dataset()
    print("dataset:", tuple(X.shape), "features |", tuple(y.shape), "labels")

    model = MLP()
    train(model, X, y)

    acc = accuracy(model, X, y)
    print(f"final training accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
