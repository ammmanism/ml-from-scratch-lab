"""
Regularization utilities for the ML From Scratch library.

Contains implementations of various regularization penalties.
"""

import numpy as np
from typing import Union


def l1_penalty(weights: np.ndarray, alpha: float = 1.0) -> float:
    """
    Compute the L1 regularization penalty (Lasso).
    
    L1 penalty = alpha * sum(|weights|)
    
    Args:
        weights: Weight vector or matrix of any shape
        alpha: Regularization strength (default: 1.0)
        
    Returns:
        L1 penalty value
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    
    return alpha * np.sum(np.abs(weights))


def l2_penalty(weights: np.ndarray, alpha: float = 1.0) -> float:
    """
    Compute the L2 regularization penalty (Ridge).
    
    L2 penalty = alpha * sum(weights^2)
    
    Args:
        weights: Weight vector or matrix of any shape
        alpha: Regularization strength (default: 1.0)
        
    Returns:
        L2 penalty value
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    
    return alpha * np.sum(weights ** 2)


def elasticnet_penalty(weights: np.ndarray, alpha: float = 1.0, l1_ratio: float = 0.5) -> float:
    """
    Compute the ElasticNet regularization penalty.
    
    ElasticNet penalty = alpha * (l1_ratio * L1_penalty + (1 - l1_ratio) * L2_penalty)
    
    Args:
        weights: Weight vector or matrix of any shape
        alpha: Regularization strength (default: 1.0)
        l1_ratio: Mixing parameter between L1 and L2 penalties (default: 0.5)
                  l1_ratio=1 corresponds to L1 penalty, l1_ratio=0 to L2 penalty
        
    Returns:
        ElasticNet penalty value
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    
    if not (0 <= l1_ratio <= 1):
        raise ValueError(f"l1_ratio must be between 0 and 1, got {l1_ratio}")
    
    l1_pen = l1_penalty(weights, alpha=alpha * l1_ratio)
    l2_pen = l2_penalty(weights, alpha=alpha * (1 - l1_ratio))
    
    return l1_pen + l2_pen