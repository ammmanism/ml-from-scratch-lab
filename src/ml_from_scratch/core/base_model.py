from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models in ml_from_scratch.
    
    Enforces a Scikit-learn like API with fit and predict methods.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Fit the model to the training data.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,) or (n_samples, n_targets).
            
        Returns:
            self: Returns an instance of self.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels or values for samples in X.
        
        Args:
            X (np.ndarray): Samples of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Predicted values of shape (n_samples,) or (n_samples, n_targets).
        """
        pass
