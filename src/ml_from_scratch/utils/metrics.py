"""Utility metrics for evaluating model performance."""

import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy classification score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score as a float between 0 and 1
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check that arrays have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    # Calculate accuracy
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    """
    Calculate mean squared error regression loss.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Mean squared error as a float
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check that arrays have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    # Calculate mean squared error
    return np.mean((y_true - y_pred) ** 2)