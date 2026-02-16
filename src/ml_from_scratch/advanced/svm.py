"""
Support Vector Machine (SVM) implementation for the ML From Scratch library.

Contains the SVM classifier implementing linear SVM with gradient descent optimization.
"""

import numpy as np
from typing import Optional
from ..utils import train_test_split


class SVM:
    """
    Support Vector Machine classifier using primal form with gradient descent.
    
    Attributes:
        learning_rate (float): Learning rate for gradient descent
        n_iterations (int): Number of training iterations
        C (float): Regularization parameter (inverse of regularization strength)
        random_state (int): Random seed for reproducibility
        weights (np.ndarray): Learned weights after fitting
        bias (float): Learned bias after fitting
    """

    def __init__(self, learning_rate: float = 0.001, n_iterations: int = 1000, C: float = 1.0,
                 random_state: Optional[int] = None):
        """
        Initialize the SVM classifier.
        
        Args:
            learning_rate: Learning rate for gradient descent (default: 0.001)
            n_iterations: Number of training iterations (default: 1000)
            C: Regularization parameter (inverse of regularization strength) (default: 1.0)
            random_state: Random seed for reproducibility (default: None)
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if n_iterations <= 0:
            raise ValueError("Number of iterations must be positive")
        if C <= 0:
            raise ValueError("C must be positive")
        
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.C = C
        self.random_state = random_state
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        """
        Fit the SVM classifier to training data.
        
        Args:
            X: Training data features of shape (n_samples, n_features)
            y: Training data targets of shape (n_samples,) with binary values (-1, 1) or (0, 1)
            
        Returns:
            Self (for method chaining)
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: X {X.shape[0]}, y {y.shape[0]}")
        
        # Convert labels to -1, 1 if they are 0, 1
        y_modified = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Initialize weights and bias
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        
        # Gradient descent optimization
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Check if sample is misclassified or within margin
                condition = y_modified[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                
                if condition:
                    # Correctly classified, update weights with only regularization
                    self.weights -= self.learning_rate * (2 * self.C * self.weights)
                else:
                    # Misclassified or within margin, update with hinge loss gradient
                    self.weights -= self.learning_rate * (2 * self.C * self.weights - 2 * y_modified[idx] * x_i)
                    self.bias -= self.learning_rate * (-2 * y_modified[idx])
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted classes of shape (n_samples,) with binary values (-1, 1) or (0, 1)
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if X.ndim != 2 or X.shape[1] != self.weights.shape[0]:
            raise ValueError(f"X must have {self.weights.shape[0]} features, got {X.shape[1]}")
        
        approximations = np.dot(X, self.weights) - self.bias
        return np.sign(approximations)