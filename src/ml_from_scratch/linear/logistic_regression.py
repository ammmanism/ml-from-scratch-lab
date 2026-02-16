"""
Logistic regression implementation for the ML From Scratch library.

Contains the LogisticRegression class with gradient descent optimization and L2 regularization.
"""

import numpy as np
from typing import Optional
from ..utils import sigmoid
from ..linear.regularization import l2_penalty


class LogisticRegression:
    """
    Logistic Regression classifier for binary classification.
    
    This implementation uses gradient descent for optimization and supports L2 regularization.
    
    Attributes:
        coefficients_ (np.ndarray): Model coefficients after fitting
        intercept_ (float): Intercept term after fitting
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, alpha: float = 0.0):
        """
        Initialize the Logistic Regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent (default: 0.01)
            n_iterations: Number of iterations for gradient descent (default: 1000)
            alpha: Regularization strength for L2 penalty (default: 0.0)
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if n_iterations <= 0:
            raise ValueError("Number of iterations must be positive")
        
        if alpha < 0:
            raise ValueError("Alpha must be non-negative")
        
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.coefficients_ = None
        self.intercept_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Fit the logistic regression model to training data.
        
        Args:
            X: Training data features of shape (n_samples, n_features)
            y: Training data targets of shape (n_samples,) with binary values (0 or 1)
            
        Returns:
            Self (for method chaining)
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: X {X.shape[0]}, y {y.shape[0]}")
        
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError("y must contain only binary values (0 and 1)")
        
        n_samples, n_features = X.shape
        
        # Initialize coefficients randomly
        self.coefficients_ = np.random.normal(0, 0.01, n_features)
        self.intercept_ = 0.0
        
        # Gradient descent loop
        for _ in range(self.n_iterations):
            # Compute linear combination
            linear_combination = X @ self.coefficients_ + self.intercept_
            
            # Apply sigmoid function to get probabilities
            y_pred_proba = sigmoid(linear_combination)
            
            # Compute gradients
            # For logistic regression with L2 regularization:
            # dJ/dw = (1/m) * X.T @ (predictions - y) + 2 * alpha * w
            # dJ/db = (1/m) * sum(predictions - y)
            
            dw = (1/n_samples) * X.T @ (y_pred_proba - y) + 2 * self.alpha * self.coefficients_
            db = (1/n_samples) * np.sum(y_pred_proba - y)
            
            # Update coefficients
            self.coefficients_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Probabilities of shape (n_samples,) for the positive class (class 1)
        """
        if self.coefficients_ is None or self.intercept_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if X.ndim != 2 or X.shape[1] != self.coefficients_.shape[0]:
            raise ValueError(f"X must have {self.coefficients_.shape[0]} features, got {X.shape[1]}")
        
        linear_combination = X @ self.coefficients_ + self.intercept_
        return sigmoid(linear_combination)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted classes of shape (n_samples,) with binary values (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)