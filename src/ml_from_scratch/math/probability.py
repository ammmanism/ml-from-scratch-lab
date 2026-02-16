"""
Probability utilities for the ML From Scratch library.

Contains implementations of probability distributions and statistical functions.
"""

import numpy as np
from typing import Tuple


def sample_gaussian(mean: np.ndarray, cov: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Sample from a multivariate Gaussian distribution.
    
    Args:
        mean: Mean vector of shape (n_features,)
        cov: Covariance matrix of shape (n_features, n_features)
        n_samples: Number of samples to generate
        
    Returns:
        Samples array of shape (n_samples, n_features)
    """
    if mean.ndim != 1:
        raise ValueError(f"Mean must be 1-dimensional, got shape {mean.shape}")
    
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance matrix must be square, got shape {cov.shape}")
    
    if mean.shape[0] != cov.shape[0]:
        raise ValueError(f"Dimension mismatch: mean {mean.shape} vs cov {cov.shape}")
    
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    
    # Use numpy's built-in multivariate normal sampling
    samples = np.random.multivariate_normal(mean, cov, size=n_samples)
    return samples


def simulate_clt(n_trials: int, sample_size: int) -> np.ndarray:
    """
    Simulate the Central Limit Theorem by sampling from a uniform distribution.
    
    The Central Limit Theorem states that the distribution of sample means approaches
    a normal distribution as the sample size increases, regardless of the population distribution.
    
    Args:
        n_trials: Number of trials/samples of means to generate
        sample_size: Size of each sample used to compute the mean
        
    Returns:
        Array of sample means of shape (n_trials,)
    """
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")
    
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    
    # Generate sample means from uniform distribution [0, 1]
    sample_means = []
    
    for _ in range(n_trials):
        # Generate sample_size random numbers from uniform distribution [0, 1]
        sample = np.random.uniform(0, 1, size=sample_size)
        # Compute the mean of the sample
        sample_mean = np.mean(sample)
        sample_means.append(sample_mean)
    
    return np.array(sample_means)