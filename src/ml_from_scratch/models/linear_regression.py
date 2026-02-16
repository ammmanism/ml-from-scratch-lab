"""Linear Regression implementation from scratch."""

import numpy as np
from ..core.base_model import BaseModel
from ..core.losses import MSELoss
from ..optimizers import SGD


class LinearRegression(BaseModel):
    """Linear Regression model implemented from scratch."""
    
    def __init__(self, fit_intercept=True):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = None
        self.loss_fn = MSELoss()
        
    def fit(self, X, y, lr=0.01, epochs=1000):
        """
        Fit the linear regression model.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
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
            y_pred = self.predict(X)
            
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
                
    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        """
        Calculate the coefficient of determination (R^2).
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: True target values of shape (n_samples,)
            
        Returns:
            R^2 score
        """
        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot)