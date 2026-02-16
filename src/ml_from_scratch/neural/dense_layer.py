"""Dense (fully connected) layer implementation."""

import numpy as np
from ..core.initializers import HeNormal


class Dense:
    """Dense (fully connected) layer."""
    
    def __init__(self, units, input_dim=None, activation=None, initializer='he_normal', bias=True):
        """
        Initialize the dense layer.
        
        Args:
            units: Number of neurons in the layer
            input_dim: Dimension of input to the layer (can be set later)
            activation: Activation function to apply
            initializer: Weight initialization method
            bias: Whether to include bias term
        """
        self.units = units
        self.input_dim = input_dim
        self.activation = activation
        self.use_bias = bias
        self.initializer = HeNormal() if initializer == 'he_normal' else initializer
        
        # Parameters to be initialized during first forward pass
        self.weights = None
        self.biases = None
        self.input = None  # Store input for backward pass
        
    def build(self, input_shape):
        """Build the layer with the specified input shape."""
        if len(input_shape) == 1:
            input_dim = input_shape[0]
        else:
            input_dim = input_shape[-1]
        
        # Initialize weights and biases
        self.weights = self.initializer.initialize((input_dim, self.units))
        if self.use_bias:
            self.biases = np.zeros((self.units,))
    
    def forward(self, inputs):
        """Forward pass through the layer."""
        if self.weights is None:
            self.build(inputs.shape)
        
        self.input = inputs  # Store input for backward pass
        z = np.dot(inputs, self.weights)
        if self.use_bias:
            z += self.biases
        
        if self.activation:
            return self.activation.forward(z)
        return z
    
    def backward(self, grad_output):
        """Backward pass through the layer."""
        if self.activation:
            grad_output = self.activation.backward(grad_output)
        
        # Gradients w.r.t. weights and biases
        dW = np.dot(self.input.T, grad_output)
        dB = np.sum(grad_output, axis=0) if self.use_bias else None
        dX = np.dot(grad_output, self.weights.T)
        
        return dX, {'weights': dW, 'biases': dB}
    
    def get_parameters(self):
        """Get layer parameters."""
        params = {'weights': self.weights}
        if self.use_bias:
            params['biases'] = self.biases
        return params
    
    def set_parameters(self, params):
        """Set layer parameters."""
        self.weights = params['weights']
        if self.use_bias and 'biases' in params:
            self.biases = params['biases']