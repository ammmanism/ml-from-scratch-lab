"""
Utility tools for the ML from Scratch library

⚠️ AI-GENERATED CODE
This module was generated using AI assistance.
Review and test thoroughly before using in production.

Author: GitHub Copilot (AI Assistant)
Created: February 2026
"""

import numpy as np
from typing import Union, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Utilities:
    """Collection of utility functions for ML operations"""
    
    @staticmethod
    def validate_array(arr: Any, name: str = "array") -> np.ndarray:
        """
        Validate and convert input to numpy array
        
        Args:
            arr: Input array-like object
            name: Name for error messages
            
        Returns:
            Validated numpy array
            
        Raises:
            ValueError: If array cannot be created
        """
        try:
            return np.array(arr)
        except Exception as e:
            raise ValueError(f"Invalid {name}: {e}")
    
    @staticmethod
    def check_shapes(*arrays: np.ndarray) -> bool:
        """
        Check compatibility of array shapes
        
        Args:
            arrays: Variable number of arrays to check
            
        Returns:
            True if shapes are compatible
            
        Raises:
            ValueError: If shapes are incompatible
        """
        if not arrays:
            return True
        
        first_shape = arrays[0].shape
        for i, arr in enumerate(arrays[1:], 1):
            if arr.shape != first_shape:
                raise ValueError(
                    f"Shape mismatch: array 0 has shape {first_shape}, "
                    f"array {i} has shape {arr.shape}"
                )
        return True
    
    @staticmethod
    def normalize(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """
        Normalize array to [0, 1] range
        
        Args:
            arr: Input array
            axis: Axis along which to normalize
            
        Returns:
            Normalized array
        """
        arr = np.array(arr, dtype=np.float64)
        min_val = np.min(arr, axis=axis, keepdims=True)
        max_val = np.max(arr, axis=axis, keepdims=True)
        
        return (arr - min_val) / (max_val - min_val + 1e-8)
    
    @staticmethod
    def standardize(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """
        Standardize array (zero mean, unit variance)
        
        Args:
            arr: Input array
            axis: Axis along which to standardize
            
        Returns:
            Standardized array
        """
        arr = np.array(arr, dtype=np.float64)
        mean = np.mean(arr, axis=axis, keepdims=True)
        std = np.std(arr, axis=axis, keepdims=True)
        
        return (arr - mean) / (std + 1e-8)
    
    @staticmethod
    def clip(arr: np.ndarray, min_val: float = -1e10, max_val: float = 1e10) -> np.ndarray:
        """
        Clip array values to range
        
        Args:
            arr: Input array
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Clipped array
        """
        return np.clip(arr, min_val, max_val)
    
    @staticmethod
    def safe_log(arr: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
        """
        Safe logarithm (handles near-zero values)
        
        Args:
            arr: Input array
            epsilon: Small value to avoid log(0)
            
        Returns:
            Logarithm of array
        """
        return np.log(np.clip(arr, epsilon, None))
    
    @staticmethod
    def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
                   epsilon: float = 1e-15) -> np.ndarray:
        """
        Safe division (handles zero denominators)
        
        Args:
            numerator: Numerator array
            denominator: Denominator array
            epsilon: Small value to prevent division by zero
            
        Returns:
            Safe division result
        """
        return numerator / (denominator + epsilon)
    
    @staticmethod
    def one_hot_encode(labels: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
        """
        One-hot encode labels
        
        Args:
            labels: Integer labels
            num_classes: Number of classes (if None, inferred from max label)
            
        Returns:
            One-hot encoded labels
        """
        labels = np.array(labels, dtype=int)
        
        if num_classes is None:
            num_classes = int(np.max(labels)) + 1
        
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        
        return one_hot
    
    @staticmethod
    def train_test_split(X: np.ndarray, y: np.ndarray, 
                        test_size: float = 0.2, 
                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion of test data (0-1)
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        split_idx = int(len(X) * (1 - test_size))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# Convenience functions
def validate_array(arr: Any, name: str = "array") -> np.ndarray:
    """Quick wrapper for array validation"""
    return Utilities.validate_array(arr, name)


def normalize(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Quick wrapper for normalization"""
    return Utilities.normalize(arr, axis)


def standardize(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Quick wrapper for standardization"""
    return Utilities.standardize(arr, axis)


__all__ = [
    'Utilities',
    'validate_array',
    'normalize',
    'standardize',
]
