"""
Linear regression implementation for the ML From Scratch library.

Contains the LinearRegression class with both Normal Equation and Gradient Descent fitting methods.
"""

import numpy as np
from typing import Optional, Union
from ..linear.regularization import l2_penalty


class LinearRegression:
    """
    Linear Regression model with support for L2 regularization.
    
    This implementation provides two fitting methods:
    1. Normal Equation (closed-form solution)
    2. Gradient Descent (iterative optimization)
    
    Attributes:
        coefficients_ (np.ndarray): Model coefficients after fitting
        intercept_ (float): Intercept term after fitting
        method (str): Fitting method used ('normal' or 'gradient_descent')
    """
    
    def __init__(self, method: str = 'normal', alpha: float = 0.0, 
                 learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        Initialize the Linear Regression model.
        
        Args:
            method: Fitting method ('normal' for Normal Equation, 'gradient_descent' for GD)
            alpha: Regularization strength for L2 penalty (default: 0.0)
            learning_rate: Learning rate for gradient descent (default: 0.01)
            n_iterations: Number of iterations for gradient descent (default: 1000)
        """
        if method not in ['normal', 'gradient_descent']:
            raise ValueError("Method must be 'normal' or 'gradient_descent'")
        
        if alpha < 0:
            raise ValueError("Alpha must be non-negative")
        
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if n_iterations <= 0:
            raise ValueError("Number of iterations must be positive")
        
        self.method = method
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coefficients_ = None
        self.intercept_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear regression model to training data.
        
        Args:
            X: Training data features of shape (n_samples, n_features)
            y: Training data targets of shape (n_samples,)
            
        Returns:
            Self (for method chaining)
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: X {X.shape[0]}, y {y.shape[0]}")
        
        # Add bias term by adding column of ones to X
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        if self.method == 'normal':
            # Normal equation: theta = (X^T * X + alpha * I)^(-1) * X^T * y
            # We exclude the bias term from regularization by creating a modified identity matrix
            reg_matrix = self.alpha * np.eye(X_with_bias.shape[1])
            reg_matrix[0, 0] = 0  # Don't regularize the bias term
            
            try:
                # Solve normal equation
                XtX_reg = X_with_bias.T @ X_with_bias + reg_matrix
                self.coefficients_ = np.linalg.solve(XtX_reg, X_with_bias.T @ y)
                
                # Extract intercept and coefficients
                self.intercept_ = self.coefficients_[0]
                self.coefficients_ = self.coefficients_[1:]
                
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudo-inverse
                XtX_reg = X_with_bias.T @ X_with_bias + reg_matrix
                coeffs = np.linalg.pinv(XtX_reg) @ X_with_bias.T @ y
                self.intercept_ = coeffs[0]
                self.coefficients_ = coeffs[1:]
        
        elif self.method == 'gradient_descent':
            # Initialize coefficients randomly
            n_features = X_with_bias.shape[1]
            theta = np.random.normal(0, 0.01, n_features)
            
            # Gradient descent loop
            for _ in range(self.n_iterations):
                # Compute predictions
                y_pred = X_with_bias @ theta
                
                # Compute gradients
                # For L2 regularization, the gradient includes the regularization term
                error = y_pred - y
                gradients = (2 / X_with_bias.shape[0]) * X_with_bias.T @ error
                
                # Add regularization term (excluding bias term)
                reg_term = np.zeros_like(theta)
                reg_term[1:] = 2 * self.alpha * theta[1:]  # Don't regularize bias
                gradients += reg_term
                
                # Update coefficients
                theta -= self.learning_rate * gradients
            
            # Extract intercept and coefficients
            self.intercept_ = theta[0]
            self.coefficients_ = theta[1:]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        if self.coefficients_ is None or self.intercept_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if X.ndim != 2 or X.shape[1] != self.coefficients_.shape[0]:
            raise ValueError(f"X must have {self.coefficients_.shape[0]} features, got {X.shape[1]}")
        
        return X @ self.coefficients_ + self.intercept_