"""
Decision Tree implementation for the ML From Scratch library.

Contains the DecisionTreeClassifier class implementing CART algorithm with Gini impurity.
"""

import numpy as np
from typing import Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class Node:
    """
    Represents a node in the decision tree.
    
    Attributes:
        feature_index: Index of feature used for splitting (None for leaf)
        threshold: Threshold value for splitting (None for leaf)
        left: Left child node
        right: Right child node
        value: Prediction value (for leaf nodes)
        is_leaf: Whether this is a leaf node
    """
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    value: Optional[float] = None
    is_leaf: bool = False


class DecisionTreeClassifier:
    """
    Decision Tree Classifier implementing the CART algorithm with Gini impurity.
    
    Attributes:
        root (Node): Root node of the decision tree
        max_depth (int): Maximum depth of the tree
        min_samples_split (int): Minimum number of samples required to split a node
        n_classes (int): Number of classes in the target variable
    """

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        """
        Initialize the Decision Tree Classifier.
        
        Args:
            max_depth: Maximum depth of the tree (default: 10)
            min_samples_split: Minimum number of samples required to split a node (default: 2)
        """
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
            
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes = None

    def _gini_impurity(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity of a set of labels.
        
        Args:
            y: Array of labels
            
        Returns:
            Gini impurity value
        """
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Find the best split for the current node based on Gini impurity.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing best split information
        """
        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_split = {'feature_index': None, 'threshold': None, 'gini': best_gini}

        # Only compute initial gini once
        current_gini = self._gini_impurity(y)

        # Iterate through all features
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)

            # Iterate through all possible thresholds
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2

                # Split the data
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate weighted Gini impurity after split
                left_y = y[left_mask]
                right_y = y[right_mask]

                n_left, n_right = len(left_y), len(right_y)
                gini_left = self._gini_impurity(left_y)
                gini_right = self._gini_impurity(right_y)

                weighted_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right

                # Update best split if this is better
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'gini': weighted_gini
                    }

        return best_split

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively build the decision tree.
        
        Args:
            X: Feature matrix
            y: Target vector
            depth: Current depth in the tree
            
        Returns:
            Node representing the subtree
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            # Create leaf node
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, is_leaf=True)

        # Find best split
        best_split = self._best_split(X, y)

        # If no valid split found, create leaf
        if best_split['feature_index'] is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, is_leaf=True)

        # Split the data
        feature_index = best_split['feature_index']
        threshold = best_split['threshold']
        
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        # Build left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Create internal node
        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left_subtree,
            right=right_subtree
        )

    def _most_common_label(self, y: np.ndarray) -> float:
        """
        Find the most common label in an array.
        
        Args:
            y: Array of labels
            
        Returns:
            Most common label
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """
        Fit the decision tree classifier to training data.
        
        Args:
            X: Training data features of shape (n_samples, n_features)
            y: Training data targets of shape (n_samples,)
            
        Returns:
            Self (for method chaining)
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: X {X.shape[0]}, y {y.shape[0]}")

        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y)
        return self

    def _predict_sample(self, x: np.ndarray, node: Node) -> float:
        """
        Predict a single sample using the tree.
        
        Args:
            x: Single sample of shape (n_features,)
            node: Current node in the tree
            
        Returns:
            Predicted class
        """
        # If it's a leaf node, return the value
        if node.is_leaf:
            return node.value

        # Navigate to the appropriate child based on feature value
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted classes of shape (n_samples,)
        """
        if self.root is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

        predictions = []
        for x in X:
            pred = self._predict_sample(x, self.root)
            predictions.append(pred)

        return np.array(predictions)