"""
Linear algebra utilities for the ML From Scratch library.

Contains implementations of fundamental linear algebra algorithms.
"""

import numpy as np
from typing import Optional


def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system of linear equations Ax = b using Gaussian elimination.
    
    Args:
        A: Coefficient matrix of shape (n, n)
        b: Right-hand side vector of shape (n,)
        
    Returns:
        Solution vector x of shape (n,) such that Ax = b
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square (n, n)")
    
    if b.ndim != 1:
        raise ValueError("Vector b must be 1-dimensional")
    
    if A.shape[0] != b.shape[0]:
        raise ValueError(f"Incompatible dimensions: A {A.shape}, b {b.shape}")
    
    n = A.shape[0]
    # Create augmented matrix [A|b]
    aug_matrix = np.column_stack([A.astype(float), b.astype(float)])
    
    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = np.argmax(np.abs(aug_matrix[i:, i])) + i
        aug_matrix[[i, max_row]] = aug_matrix[[max_row, i]]
        
        # Check for singular matrix
        if abs(aug_matrix[i, i]) < 1e-10:
            raise ValueError("Matrix is singular or nearly singular")
        
        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
            factor = aug_matrix[k, i] / aug_matrix[i, i]
            aug_matrix[k, i:] -= factor * aug_matrix[i, i:]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (aug_matrix[i, -1] - np.dot(aug_matrix[i, i + 1:n], x[i + 1:n])) / aug_matrix[i, i]
    
    return x


def power_iteration(A: np.ndarray, n_iter: int = 1000, tol: float = 1e-6) -> tuple:
    """
    Compute the dominant eigenvalue and corresponding eigenvector of matrix A using power iteration.
    
    Args:
        A: Square matrix of shape (n, n)
        n_iter: Number of iterations (default: 1000)
        tol: Tolerance for convergence (default: 1e-6)
        
    Returns:
        Tuple of (eigenvalue, eigenvector) where eigenvalue is the dominant eigenvalue
        and eigenvector is the corresponding normalized eigenvector
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square (n, n)")
    
    n = A.shape[0]
    # Initialize random vector
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(n_iter):
        # Multiply by A
        w = A @ v
        
        # Compute eigenvalue estimate
        eigenval = v.T @ w
        
        # Normalize
        v_new = w / np.linalg.norm(w)
        
        # Check for convergence
        if np.linalg.norm(v_new - v) < tol:
            v = v_new
            break
            
        v = v_new
    
    # Final computation of eigenvalue using Rayleigh quotient
    Av = A @ v
    eigenval = (v.T @ Av) / (v.T @ v)
    
    return eigenval, v