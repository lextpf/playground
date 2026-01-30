"""Learning goal: get comfortable with PyTorch tensors and autograd.

I want to (1) create tensors a few different ways and poke at shapes/ops,
(2) confirm the numpy <-> tensor bridge shares memory, and (3) trust autograd
by checking a computed gradient against a derivative I work out by hand.
Everything runs on CPU; CUDA is only used if it happens to be available.
"""

import numpy as np
import torch


def pick_device() -> torch.device:
    """Prefer CUDA when present, but fall back to CPU so this always runs."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_basics() -> None:
    """Creation, shapes, a couple of ops, and the numpy interop quirk."""
    # A few common construction patterns.
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    zeros = torch.zeros(2, 3)
    ones = torch.ones_like(a)
    rand = torch.rand(2, 3)  # uniform [0, 1), driven by the global seed

    print("a =\n", a)
    print("a.shape:", tuple(a.shape), "| dtype:", a.dtype, "| ndim:", a.ndim)
    print("zeros + ones =\n", zeros + ones)

    # Elementwise vs. matrix multiply, easy to mix these up early on.
    print("elementwise a * ones:\n", a * ones)
    print("matmul a @ a.T:\n", a @ a.T)

    # Reductions and reshaping.
    print("a.sum():", a.sum().item(), "| a.mean(dim=0):", a.mean(dim=0).tolist())
    print("a.view(3, 2):\n", a.view(3, 2))
    print("rand sample:\n", rand)

    # numpy bridge: torch.from_numpy shares the underlying buffer (no copy),
    # so mutating one mutates the other. Worth seeing once.
    np_arr = np.arange(4, dtype=np.float32)
    shared = torch.from_numpy(np_arr)
    np_arr[0] = 99.0
    print("shared memory after numpy edit:", shared.tolist())
    print("back to numpy:", shared.numpy().tolist())


def autograd_check(device: torch.device) -> None:
    """Differentiate f(x) = sum(x^2 + 3x) and compare to the hand derivative.

    By hand: d/dx (x^2 + 3x) = 2x + 3, applied elementwise.
    """
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)

    # f is scalar (sum over the vector), which is what .backward() expects.
    f = (x**2 + 3 * x).sum()
    f.backward()

    expected = 2 * x + 3  # the analytic gradient, computed without autograd
    print("\nf(x) =", f.item())
    print("autograd grad:", x.grad.tolist())
    print("hand-derived 2x + 3:", expected.detach().tolist())
    assert torch.allclose(x.grad, expected.detach()), "gradient mismatch!"
    print("gradients match -> autograd verified")


def main() -> None:
    np.random.seed(0)
    torch.manual_seed(0)

    device = pick_device()
    print("Using device:", device)

    tensor_basics()
    autograd_check(device)


if __name__ == "__main__":
    main()
