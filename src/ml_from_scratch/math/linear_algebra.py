"""
Core Linear Algebra Utilities for ML from Scratch.
Implements Gaussian elimination, power iteration, and SVD from first principles.
"""

import numpy as np
from typing import Tuple

def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the linear system Ax = b using Gaussian elimination with partial pivoting.

    Args:
        A: Square coefficient matrix of shape (n, n)
        b: Constant vector of shape (n,)

    Returns:
        x: Solution vector of shape (n,)

    Raises:
        ValueError: If matrix is singular or dimensions mismatch.
    """
    A = np.array(A, dtype=float).copy()
    b = np.array(b, dtype=float).copy()

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Dimensions of A and b do not match.")

    n = A.shape[0]

    # Forward elimination with partial pivoting
    for i in range(n):
        # Pivot selection
        max_idx = i + np.argmax(np.abs(A[i:, i]))
        if max_idx != i:
            A[[i, max_idx]] = A[[max_idx, i]]
            b[[i, max_idx]] = b[[max_idx, i]]

        if np.isclose(A[i, i], 0.0):
            raise ValueError("Matrix is singular or nearly singular; no unique solution.")

        # Eliminate column below pivot
        pivot = A[i, i]
        for j in range(i + 1, n):
            factor = A[j, i] / pivot
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x


def power_iteration(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-7) -> Tuple[float, np.ndarray]:
    """
    Find the dominant eigenvalue and corresponding eigenvector using power iteration.

    Args:
        A: Square matrix of shape (n, n)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance for eigenvalue change

    Returns:
        eigenvalue: Dominant eigenvalue
        eigenvector: Corresponding unit eigenvector
    """
    A = np.array(A, dtype=float).copy()
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square for eigenvalue decomposition.")

    n = A.shape[0]
    rng = np.random.RandomState(42)
    v = rng.rand(n)
    v /= np.linalg.norm(v)

    eigenvalue = 0.0
    for _ in range(max_iter):
        w = A @ v
        new_eigenvalue = np.dot(v, w)  # Rayleigh quotient
        v = w / np.linalg.norm(w)

        if np.abs(new_eigenvalue - eigenvalue) < tol:
            break
        eigenvalue = new_eigenvalue

    # Canonicalize sign: largest magnitude component positive
    if v[np.argmax(np.abs(v))] < 0:
        v *= -1.0

    return eigenvalue, v


def svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the economy Singular Value Decomposition: A = U @ diag(S) @ Vt.
    Uses eigendecomposition of A^T A. Suitable for educational/ML contexts.

    Args:
        A: Input matrix of shape (m, n)

    Returns:
        U: Left singular vectors, shape (m, k)
        S: Singular values in descending order, shape (k,)
        Vt: Right singular vectors (transposed), shape (k, n)
        where k = min(m, n)
    """
    A = np.array(A, dtype=float).copy()
    m, n = A.shape
    k = min(m, n)

    # Compute V and singular values from A^T A
    ATA = A.T @ A
    eig_vals, V = np.linalg.eigh(ATA)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    V = V[:, idx]
    Vt = V.T[:k]

    # Singular values (clamp negatives to 0 for float stability)
    S = np.sqrt(np.maximum(eig_vals, 0.0))[:k]

    # Compute U = A V S^{-1}
    U = np.zeros((m, k))
    for i in range(k):
        if S[i] > 1e-10:
            U[:, i] = A @ Vt[i] / S[i]

    return U, S, Vt


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("🔹 Linear Algebra Utilities Demo\n" + "="*40)

    # 1. Gaussian Elimination
    print("\n📐 1. Gaussian Elimination")
    A_sys = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
    b_sys = np.array([8, -11, -3])
    x_sol = gaussian_elimination(A_sys, b_sys)
    print(f"   A =\n{A_sys}")
    print(f"   b = {b_sys}")
    print(f"   ✅ Solution x = {x_sol}")
    print(f"   ✅ Residual ||Ax - b|| = {np.linalg.norm(A_sys @ x_sol - b_sys):.2e}")

    # 2. Power Iteration
    print("\n⚡ 2. Power Iteration")
    A_eig = np.array([[4, 1], [2, 3]])
    lam, v_eig = power_iteration(A_eig)
    true_lam, true_v = np.linalg.eig(A_eig)
    idx = np.argmax(np.abs(true_lam))
    print(f"   A =\n{A_eig}")
    print(f"   ✅ Dominant eigenvalue (ours): {lam:.4f} | (numpy ref): {true_lam[idx]:.4f}")
    print(f"   ✅ Eigenvector alignment: {np.abs(np.dot(v_eig, true_v[:, idx])):.4f} (should be ~1.0)")

    # 3. SVD
    print("\n📊 3. SVD (Economy)")
    A_svd = np.array([[3, 2, 2], [2, 3, -2]])
    U, S, Vt = svd(A_svd)
    A_reconstructed = U @ np.diag(S) @ Vt
    print(f"   Original A shape: {A_svd.shape} | Reconstructed error: {np.linalg.norm(A_svd - A_reconstructed):.2e}")
    print(f"   U shape: {U.shape} | S shape: {S.shape} | Vt shape: {Vt.shape}")
    print(f"   Singular values: {S}")