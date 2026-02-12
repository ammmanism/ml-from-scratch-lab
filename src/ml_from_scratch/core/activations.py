import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
    """
    Abstract base class for activation functions.
    """
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the forward pass."""
        pass

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Compute the backward pass (derivative)."""
        pass

class ReLU(Activation):
    """
    Rectified Linear Unit activation function.
    f(x) = max(0, x)
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU:
        1 if x > 0
        0 if x <= 0
        """
        return np.where(x > 0, 1, 0)

class Sigmoid(Activation):
    """
    Sigmoid activation function.
    f(x) = 1 / (1 + exp(-x))
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of Sigmoid:
        f'(x) = f(x) * (1 - f(x))
        """
        s = self.forward(x)
        return s * (1 - s)

class Tanh(Activation):
    """
    Hyperbolic Tangent activation function.
    f(x) = tanh(x)
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of Tanh:
        f'(x) = 1 - f(x)^2
        """
        return 1 - np.tanh(x)**2

class Softmax(Activation):
    """
    Softmax activation function.
    Stable implementation.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of Softmax is complex (Jacobian matrix).
        Usually handled combined with CrossEntropy.
        This provides the Jacobian per sample.
        """
        s = self.forward(x)
        # Create Jacobian matrix for each sample
        # This is often too expensive to use explicitly in simple backprop.
        # Placeholder for completeness or specific implementation needs.
        raise NotImplementedError("Softmax backward is usually handled with CrossEntropyLoss for efficiency.")
