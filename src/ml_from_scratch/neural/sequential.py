"""Sequential model implementation."""

import numpy as np
from collections import OrderedDict


class Sequential:
    """Sequential model - a linear stack of layers."""
    
    def __init__(self):
        self.layers = []
        self.built = False
    
    def add(self, layer):
        """Add a layer to the model."""
        self.layers.append(layer)
    
    def build(self, input_shape):
        """Build the model with the specified input shape."""
        prev_shape = input_shape
        for layer in self.layers:
            layer.build(prev_shape)
            # Determine output shape of this layer to feed to next layer
            if hasattr(layer, 'units'):
                prev_shape = (layer.units,)
            elif hasattr(layer, 'output_shape'):
                prev_shape = layer.output_shape
            else:
                # For activation functions, shape stays the same
                prev_shape = prev_shape
        self.built = True
    
    def compile(self, optimizer, loss):
        """Compile the model with optimizer and loss function."""
        self.optimizer = optimizer
        self.loss_fn = loss
    
    def forward(self, inputs):
        """Forward pass through all layers."""
        if not self.built:
            self.build(inputs.shape[1:])
        
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def predict(self, X):
        """Make predictions on input data."""
        return self.forward(X)
    
    def fit(self, X, y, epochs=1, batch_size=32, verbose=True):
        """Train the model on input data."""
        if not hasattr(self, 'optimizer') or not hasattr(self, 'loss_fn'):
            raise ValueError("Model must be compiled with optimizer and loss before training")
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, n_samples, batch_size):
                # Get batch
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                loss = self.loss_fn(y_batch, y_pred)
                epoch_loss += loss
                num_batches += 1
                
                # Compute gradients using backpropagation
                # Start with gradient of loss w.r.t. output
                grad = self.loss_fn.gradient(y_batch, y_pred)
                
                # Backpropagate through layers
                layer_grads = OrderedDict()
                for layer in reversed(self.layers):
                    grad, layer_param_grads = layer.backward(grad)
                    # Accumulate parameter gradients
                    for param_name, param_grad in layer_param_grads.items():
                        if param_grad is not None:  # Skip None gradients (like for bias when use_bias=False)
                            full_param_name = f"{id(layer)}_{param_name}"
                            layer_grads[full_param_name] = param_grad
                
                # Collect all parameters
                all_params = {}
                for layer in self.layers:
                    layer_params = layer.get_parameters()
                    for param_name, param_value in layer_params.items():
                        full_param_name = f"{id(layer)}_{param_name}"
                        all_params[full_param_name] = param_value
                
                # Update parameters
                updated_params = self.optimizer.update(all_params, layer_grads)
                
                # Set updated parameters back to layers
                for layer in self.layers:
                    old_params = layer.get_parameters()
                    new_params = {}
                    for param_name, param_value in old_params.items():
                        full_param_name = f"{id(layer)}_{param_name}"
                        if full_param_name in updated_params:
                            new_params[param_name] = updated_params[full_param_name]
                        else:
                            new_params[param_name] = param_value
                    layer.set_parameters(new_params)
            
            if verbose:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")