"""Data utility functions."""

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.
    
    Args:
        X: Features array of shape (n_samples, n_features)
        y: Target values array of shape (n_samples,)
        test_size: Proportion of the dataset to include in the test split
        random_state: Determines random number generation for dataset shuffling
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Check that X and y have the same number of samples
    n_samples = X.shape[0]
    if y.shape[0] != n_samples:
        raise ValueError("X and y must have the same number of samples")
    
    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Calculate split point
    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split the data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test