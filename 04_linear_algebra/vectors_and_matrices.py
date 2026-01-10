"""Learning goal: build intuition for the core vector/matrix operations that
show up everywhere in ML, dot products, norms, cosine similarity, and the
different flavors of matrix multiplication, and sanity-check them against
NumPy's reference implementations.
"""

import numpy as np


def dot(a, b):
    """Inner product of two 1-D vectors."""
    return float(np.sum(a * b))


def l1_norm(v):
    """L1 (Manhattan) norm: sum of absolute values."""
    return float(np.sum(np.abs(v)))


def l2_norm(v):
    """L2 (Euclidean) norm: sqrt of the dot product with itself."""
    return float(np.sqrt(dot(v, v)))


def cosine_similarity(a, b):
    """Cosine of the angle between two vectors (1 = aligned, 0 = orthogonal)."""
    return dot(a, b) / (l2_norm(a) * l2_norm(b))


def matvec(A, x):
    """Matrix-vector product: each output entry is a row dotted with x."""
    return np.array([dot(row, x) for row in A])


def matmul(A, B):
    """Matrix-matrix product computed via row/column dot products."""
    B_cols = B.T
    return np.array([[dot(row, col) for col in B_cols] for row in A])


def transpose(A):
    """Swap rows and columns."""
    return A.T


def identity(n):
    """n x n identity matrix."""
    return np.eye(n)


def main():
    np.random.seed(0)

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, -5.0, 6.0])

    # Verify our hand-rolled ops against NumPy's reference implementations.
    assert np.isclose(dot(a, b), np.dot(a, b))
    assert np.isclose(l2_norm(a), np.linalg.norm(a))
    assert np.isclose(l1_norm(a), np.linalg.norm(a, ord=1))
    print(f"dot(a, b)        = {dot(a, b):.4f}")
    print(f"L1 norm of a     = {l1_norm(a):.4f}")
    print(f"L2 norm of a     = {l2_norm(a):.4f}")
    print(f"cosine(a, b)     = {cosine_similarity(a, b):.4f}")

    A = np.random.randn(3, 4)
    B = np.random.randn(4, 2)
    x = np.random.randn(4)

    # Matrix-vector and matrix-matrix products vs the @ operator.
    assert np.allclose(matvec(A, x), A @ x)
    assert np.allclose(matmul(A, B), A @ B)
    print(f"A @ x close to NumPy? {np.allclose(matvec(A, x), A @ x)}")
    print(f"A @ B close to NumPy? {np.allclose(matmul(A, B), A @ B)}")

    # Transpose flips the shape; identity leaves a matrix unchanged.
    assert transpose(A).shape == (4, 3)
    assert np.allclose(A @ identity(4), A)
    print(f"A shape {A.shape} -> A.T shape {transpose(A).shape}")
    print("All linear algebra checks passed.")


if __name__ == "__main__":
    main()
