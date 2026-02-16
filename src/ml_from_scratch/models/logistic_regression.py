"""Logistic Regression implementation from scratch."""

import numpy as np
from ..core.base_model import BaseModel
from ..core.losses import BinaryCrossEntropy
from ..optimizers import SGD


def sigmoid(x):
    """Sigmoid activation function."""
    # Clip x to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


class LogisticRegression(BaseModel):
    """Logistic Regression model implemented from scratch."""
    
    def __init__(self, fit_intercept=True):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = None
        self.loss_fn = BinaryCrossEntropy()
        
    def fit(self, X, y, lr=0.01, epochs=1000):
        """
        Fit the logistic regression model.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Binary target values of shape (n_samples,) with values 0 or 1
            lr: Learning rate for gradient descent
            epochs: Number of training epochs
        """
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.random.normal(0, 0.01, size=(n_features,))
        self.bias = 0.0 if self.fit_intercept else 0.0
        
        # Initialize optimizer
        optimizer = SGD(learning_rate=lr)
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.predict_proba(X)
            
            # Compute loss
            loss = self.loss_fn(y, y_pred)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y) if self.fit_intercept else 0.0
            
            # Prepare parameters and gradients for optimizer
            params = {'weights': self.weights}
            grads = {'weights': dw}
            
            if self.fit_intercept:
                params['bias'] = self.bias
                grads['bias'] = db
            
            # Update parameters
            updated_params = optimizer.update(params, grads)
            self.weights = updated_params['weights']
            
            if self.fit_intercept:
                self.bias = updated_params['bias']
                
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Probabilities of shape (n_samples,) for the positive class
        """
        X = np.array(X)
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)
    
    def predict(self, X):
        """
        Make binary predictions.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Binary predictions of shape (n_samples,) with values 0 or 1
        """
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def score(self, X, y):
        """
        Calculate the accuracy score.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: True binary target values of shape (n_samples,) with values 0 or 1
            
        Returns:
            Accuracy score
        """
        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        return np.mean(y == y_pred)