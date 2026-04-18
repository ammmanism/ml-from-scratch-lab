"""
Utility functions for the ML From Scratch library.

Contains common mathematical functions, evaluation metrics, and data processing utilities.
"""

import numpy as np
from typing import Tuple, Optional, Union


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.
    
    The sigmoid function is defined as: sigmoid(x) = 1 / (1 + exp(-x))
    To ensure numerical stability, we clip x values to avoid overflow.
    
    Args:
        x: Input array of any shape
        
    Returns:
        Output array of same shape as input with sigmoid applied element-wise
    """
    # Clip x to prevent overflow in exp function
    x_clipped = np.clip(x, -500, 500)
    # For numerical stability, use different formulas for positive and negative inputs
    pos_mask = (x_clipped >= 0)
    neg_mask = (x_clipped < 0)
    
    result = np.zeros_like(x_clipped)
    result[pos_mask] = 1 / (1 + np.exp(-x_clipped[pos_mask]))
    result[neg_mask] = np.exp(x_clipped[neg_mask]) / (1 + np.exp(x_clipped[neg_mask]))
    
    return result


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Compute the softmax function along the specified axis.
    
    The softmax function is defined as: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    To ensure numerical stability, we subtract the max value from each row.
    
    Args:
        x: Input array of any shape
        axis: Axis along which to compute softmax (default: 1)
        
    Returns:
        Output array of same shape as input with softmax applied along specified axis
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy score between true and predicted labels.
    
    Accuracy is defined as the fraction of correct predictions over total predictions.
    
    Args:
        y_true: True labels array of shape (n_samples,)
        y_pred: Predicted labels array of shape (n_samples,)
        
    Returns:
        Accuracy score as a float between 0 and 1
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    if len(y_true) == 0:
        return 0.0
    
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    
    return correct_predictions / total_predictions


def train_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.2, 
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays into random train and test subsets.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target array of shape (n_samples,) or (n_samples, n_targets)
        test_size: Proportion of dataset to include in test split (default: 0.2)
        random_state: Random seed for reproducibility (default: None)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Number of samples mismatch: X {X.shape[0]} vs y {y.shape[0]}")
    
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(np.ceil(test_size * n_samples))
    n_train = n_samples - n_test
    
    # Generate random indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test