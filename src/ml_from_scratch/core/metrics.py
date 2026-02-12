import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the accuracy classification score.
    
    Args:
        y_true (np.ndarray): Ground truth labels (1D array).
        y_pred (np.ndarray): Predicted labels (1D array).
        
    Returns:
        float: Accuracy score.
    """
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        # One-hot to labels
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        # Probabilities to labels
        y_pred = np.argmax(y_pred, axis=1)
        
    accuracy = np.mean(y_true == y_pred)
    return float(accuracy)

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the mean squared error regression loss.
    
    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        
    Returns:
        float: Mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)
