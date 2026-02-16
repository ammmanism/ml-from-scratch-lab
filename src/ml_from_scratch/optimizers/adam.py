"""Adam optimizer."""

import numpy as np
from .optimizer_base import Optimizer


class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
    
    def update(self, params, grads):
        """Update parameters using Adam optimizer."""
        self.t += 1
        updated_params = {}
        
        for param_name, param_value in params.items():
            if param_name in grads:
                grad = grads[param_name]
                
                if param_name not in self.m:
                    self.m[param_name] = np.zeros_like(param_value)
                    self.v[param_name] = np.zeros_like(param_value)
                
                # Update biased first moment estimate
                self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
                
                # Update biased second raw moment estimate
                self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
                
                # Update parameters
                updated_params[param_name] = (
                    param_value - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                )
            else:
                updated_params[param_name] = param_value
        
        return updated_params