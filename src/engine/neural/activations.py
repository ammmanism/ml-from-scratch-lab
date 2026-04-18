"""
Activation functions for neural networks in the ML From Scratch library.

Contains implementations of common activation functions and their derivatives.
"""

import numpy as np
from typing import Union
from ..utils import sigmoid as utils_sigmoid
from ..utils import softmax as utils_softmax


def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function.
    
    Args:
        x: Input array of any shape
        
    Returns:
        Output array of same shape as input with ReLU applied element-wise
    """
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the ReLU activation function.
    
    Args:
        x: Input array of any shape
        
    Returns:
        Derivative of ReLU applied element-wise
    """
    return (x > 0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
    
    Args:
        x: Input array of any shape
        
    Returns:
        Output array of same shape as input with sigmoid applied element-wise
    """
    return utils_sigmoid(x)


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the sigmoid activation function.
    
    Args:
        x: Input array of any shape
        
    Returns:
        Derivative of sigmoid applied element-wise
    """
    sig = utils_sigmoid(x)
    return sig * (1 - sig)


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent (tanh) activation function.
    
    Args:
        x: Input array of any shape
        
    Returns:
        Output array of same shape as input with tanh applied element-wise
    """
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the hyperbolic tangent (tanh) activation function.
    
    Args:
        x: Input array of any shape
        
    Returns:
        Derivative of tanh applied element-wise
    """
    th = np.tanh(x)
    return 1 - th ** 2


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Softmax activation function.
    
    Args:
        x: Input array of any shape
        axis: Axis along which to compute softmax (default: 1)
        
    Returns:
        Output array of same shape as input with softmax applied along specified axis
    """
    return utils_softmax(x, axis=axis)