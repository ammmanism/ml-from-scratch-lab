"""Stochastic Gradient Descent optimizer."""

import numpy as np
from .optimizer_base import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        """Update parameters using gradient descent."""
        updated_params = {}
        for param_name, param_value in params.items():
            if param_name in grads:
                updated_params[param_name] = param_value - self.learning_rate * grads[param_name]
            else:
                updated_params[param_name] = param_value
        return updated_params