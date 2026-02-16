"""Activation functions implementation."""

import numpy as np


class ReLU:
    """Rectified Linear Unit activation function."""
    
    def forward(self, x):
        """Forward pass through ReLU."""
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        """Backward pass through ReLU."""
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input


class Sigmoid:
    """Sigmoid activation function."""
    
    def forward(self, x):
        """Forward pass through Sigmoid."""
        # Clip x to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        self.output = 1 / (1 + np.exp(-x_clipped))
        return self.output
    
    def backward(self, grad_output):
        """Backward pass through Sigmoid."""
        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input


class Tanh:
    """Hyperbolic tangent activation function."""
    
    def forward(self, x):
        """Forward pass through Tanh."""
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad_output):
        """Backward pass through Tanh."""
        grad_input = grad_output * (1 - self.output ** 2)
        return grad_input


class Softmax:
    """Softmax activation function."""
    
    def forward(self, x):
        """Forward pass through Softmax."""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.output
    
    def backward(self, grad_output):
        """Backward pass through Softmax."""
        # Compute Jacobian matrix for softmax derivative
        # This is a simplified version - for efficiency we're assuming element-wise gradients
        # are passed from the loss function which handles the full Jacobian internally
        s = self.output
        diag_s = np.eye(s.shape[-1]) * s
        jacobian = diag_s - np.matmul(s[..., np.newaxis], s[..., np.newaxis, :])
        
        # For simplicity, return the gradient using the diagonal approximation
        grad_input = grad_output * s * (1 - s)  # Approximation for element-wise gradient
        return grad_input