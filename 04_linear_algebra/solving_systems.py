"""Learning goal: solve a linear system Ax = b with NumPy, then explore the
structure behind it, the inverse, the determinant, and the eigenvalue
decomposition of a symmetric matrix (the form we hit constantly in ML, e.g.
covariance matrices and PCA).
"""

import numpy as np


def solve_system(A, b):
    """Solve Ax = b for x using NumPy's direct solver."""
    return np.linalg.solve(A, b)


def inverse_and_det(A):
    """Return (A_inverse, det(A))."""
    return np.linalg.inv(A), np.linalg.det(A)


def symmetric_eig(A):
    """Eigen-decomposition for a symmetric matrix (real eigenpairs)."""
    return np.linalg.eigh(A)


def main():
    np.random.seed(0)

    A = np.array([[3.0, 2.0, -1.0],
                  [2.0, -2.0, 4.0],
                  [-1.0, 0.5, -1.0]])
    b = np.array([1.0, -2.0, 0.0])

    # Solve and confirm the solution actually satisfies the system.
    x = solve_system(A, b)
    assert np.allclose(A @ x, b)
    print(f"solution x      = {np.round(x, 4)}")
    print(f"A @ x close to b? {np.allclose(A @ x, b)}")

    A_inv, det = inverse_and_det(A)
    assert np.allclose(A @ A_inv, np.eye(3))
    print(f"det(A)          = {det:.4f}")
    print(f"A @ A_inv = I?    {np.allclose(A @ A_inv, np.eye(3))}")

    # Build a symmetric matrix so eigenvalues/eigenvectors are real-valued.
    S = A @ A.T
    eigvals, eigvecs = symmetric_eig(S)
    print(f"eigenvalues     = {np.round(eigvals, 4)}")

    # Verify the defining relation S v = lambda v for the first eigenpair.
    lam, v = eigvals[0], eigvecs[:, 0]
    assert np.allclose(S @ v, lam * v)
    print(f"S v = lambda v?   {np.allclose(S @ v, lam * v)}")


if __name__ == "__main__":
    main()
