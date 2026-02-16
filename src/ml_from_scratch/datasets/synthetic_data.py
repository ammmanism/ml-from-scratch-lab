"""Synthetic dataset generation utilities."""

import numpy as np


def make_regression(n_samples=100, n_features=1, noise=0.0, random_state=None):
    """
    Generate a random regression problem.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Standard deviation of the Gaussian noise applied to the output
        random_state: Determines random number generation for dataset creation
        
    Returns:
        X: Features array of shape (n_samples, n_features)
        y: Target values array of shape (n_samples,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate random coefficients for the linear combination
    coef = np.random.randn(n_features)
    
    # Compute target values as linear combination plus some noise
    y = np.dot(X, coef)
    
    # Add noise
    if noise > 0.0:
        y += np.random.normal(scale=noise, size=y.shape)
    
    return X, y


def make_classification(n_samples=100, n_features=2, n_classes=2, random_state=None):
    """
    Generate a random classification problem.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes (currently supports 2 classes)
        random_state: Determines random number generation for dataset creation
        
    Returns:
        X: Features array of shape (n_samples, n_features)
        y: Target values array of shape (n_samples,) with values 0 or 1
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if n_classes != 2:
        raise ValueError("Currently only binary classification is supported")
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Create a random hyperplane to separate the two classes
    weights = np.random.randn(n_features)
    bias = np.random.randn()
    
    # Compute the decision boundary
    y = (np.dot(X, weights) + bias > 0).astype(int)
    
    return X, y