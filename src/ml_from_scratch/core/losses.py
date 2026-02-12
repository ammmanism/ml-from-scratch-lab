from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    """
    Abstract base class for loss functions.
    """
    
    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the loss value.
        
        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
            
        Returns:
            float: The loss value.
        """
        pass

    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to predictions.
        
        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
            
        Returns:
            np.ndarray: Gradient of the loss with respect to y_pred.
        """
        pass

class MSE(Loss):
    """
    Mean Squared Error loss function.
    """
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the Mean Squared Error.
        
        Formula: (1/n) * sum((y_true - y_pred)^2)
        """
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of MSE.
        
        Gradient: 2 * (y_pred - y_true) / n
        """
        n = y_true.size
        return 2 * (y_pred - y_true) / n

class CrossEntropy(Loss):
    """
    Categorical Cross-Entropy loss function.
    Assumes one-hot encoded targets and Softmax probabilities.
    """
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes Cross-Entropy loss.
        
        Formula: -sum(y_true * log(y_pred)) / n
        """
        epsilon = 1e-15
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        n = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / n
        return loss

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of Cross-Entropy loss.
        """
        # Note: This is simplified; usually combined with Softmax for numerical stability.
        # But strictly for the loss separate from activation:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        n = y_true.shape[0]
        return - (y_true / y_pred) / n
