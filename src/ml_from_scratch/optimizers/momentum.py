"""Momentum optimizer."""

import numpy as np
from .optimizer_base import Optimizer


class Momentum(Optimizer):
    """Momentum optimizer."""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, params, grads):
        """Update parameters using momentum."""
        updated_params = {}
        for param_name, param_value in params.items():
            if param_name in grads:
                grad = grads[param_name]
                if param_name not in self.velocities:
                    self.velocities[param_name] = np.zeros_like(grad)
                
                self.velocities[param_name] = (
                    self.momentum * self.velocities[param_name] + 
                    self.learning_rate * grad
                )
                updated_params[param_name] = (
                    param_value - self.velocities[param_name]
                )
            else:
                updated_params[param_name] = param_value
        return updated_params