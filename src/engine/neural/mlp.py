"""
Multi-Layer Perceptron (MLP) implementation for the ML From Scratch library.

Contains the MLPClassifier class implementing fully connected neural networks.
"""

import numpy as np
from typing import Optional, List, Union
from ..utils import accuracy
from .activations import relu, relu_derivative, sigmoid, sigmoid_derivative, softmax


class MLPClassifier:
    """
    Multi-layer Perceptron classifier for neural networks.
    
    Attributes:
        hidden_layers (List[int]): Number of neurons in each hidden layer
        learning_rate (float): Learning rate for gradient descent
        n_iterations (int): Number of training iterations
        alpha (float): Regularization strength (L2 penalty)
        random_state (int): Random seed for reproducibility
        weights (List[np.ndarray]): List of weight matrices for each layer
        biases (List[np.ndarray]): List of bias vectors for each layer
        n_features (int): Number of input features
        n_classes (int): Number of output classes
    """

    def __init__(self, hidden_layers: List[int] = [100], learning_rate: float = 0.01,
                 n_iterations: int = 1000, alpha: float = 0.01, random_state: Optional[int] = None):
        """
        Initialize the MLP classifier.
        
        Args:
            hidden_layers: List of integers specifying the number of neurons in each hidden layer (default: [100])
            learning_rate: Learning rate for gradient descent (default: 0.01)
            n_iterations: Number of training iterations (default: 1000)
            alpha: Regularization strength (L2 penalty) (default: 0.01)
            random_state: Random seed for reproducibility (default: None)
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if n_iterations <= 0:
            raise ValueError("Number of iterations must be positive")
        if alpha < 0:
            raise ValueError("Alpha must be non-negative")
        
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.random_state = random_state
        self.weights = []
        self.biases = []
        self.n_features = None
        self.n_classes = None

    def _init_weights(self, layer_sizes: List[int]):
        """
        Initialize weights and biases using He initialization for ReLU networks.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1_size, ..., output_size]
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU activation
            std_dev = np.sqrt(2.0 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * std_dev
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(W)
            self.biases.append(b)

    def _forward_pass(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Perform forward propagation through the network.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            List of activation values for each layer
        """
        activations = [X]
        
        # Forward through hidden layers (using ReLU)
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = relu(z)
            activations.append(a)
        
        # Output layer (using softmax for classification)
        z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a_output = softmax(z_output, axis=1)
        activations.append(a_output)
        
        return activations

    def _backward_pass(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray]) -> tuple:
        """
        Perform backward propagation to compute gradients.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            y: True labels of shape (n_samples,) or (n_samples, n_classes)
            activations: List of activation values for each layer
            
        Returns:
            Gradients for weights and biases
        """
        m = X.shape[0]  # Number of samples
        
        # Convert y to one-hot encoding if needed
        if y.ndim == 1:
            y_onehot = np.eye(self.n_classes)[y]
        else:
            y_onehot = y
        
        # Compute output layer error
        delta = activations[-1] - y_onehot  # derivative of cross-entropy loss w.r.t. logits
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        dB = [np.zeros_like(b) for b in self.biases]
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            # Compute gradients for current layer
            dW[i] = (1/m) * np.dot(activations[i].T, delta) + 2 * self.alpha * self.weights[i]
            dB[i] = (1/m) * np.sum(delta, axis=0, keepdims=True)
            
            if i > 0:  # Not the input layer
                # Propagate error to previous layer (before activation)
                delta = np.dot(delta, self.weights[i].T) * relu_derivative(
                    np.dot(activations[i-1], self.weights[i-1]) + self.biases[i-1]
                )
        
        return dW, dB

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPClassifier':
        """
        Fit the MLP classifier to training data.
        
        Args:
            X: Training data features of shape (n_samples, n_features)
            y: Training data targets of shape (n_samples,) with integer class labels
            
        Returns:
            Self (for method chaining)
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: X {X.shape[0]}, y {y.shape[0]}")

        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        
        # Prepare layer sizes: input -> hidden layers -> output
        layer_sizes = [self.n_features] + self.hidden_layers + [self.n_classes]
        
        # Initialize weights and biases
        self._init_weights(layer_sizes)
        
        # Training loop
        for iteration in range(self.n_iterations):
            # Forward pass
            activations = self._forward_pass(X)
            
            # Backward pass
            dW, dB = self._backward_pass(X, y, activations)
            
            # Update weights and biases
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dW[i]
                self.biases[i] -= self.learning_rate * dB[i]
        
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Probabilities of shape (n_samples, n_classes) for each class
        """
        if not self.weights:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have {self.n_features} features, got {X.shape[1]}")
        
        # Forward pass to get probabilities
        activations = self._forward_pass(X)
        return activations[-1]  # Return output layer activations (probabilities)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted classes of shape (n_samples,) with integer class labels
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)