"""
K-Means clustering implementation for the ML From Scratch library.

Contains the KMeans class implementing the classic K-Means algorithm.
"""

import numpy as np
from typing import Optional, Union
from ..utils import euclidean_distance


class KMeans:
    """
    K-Means clustering algorithm.
    
    Attributes:
        n_clusters (int): Number of clusters
        init (str): Initialization method ('random' or 'k-means++')
        max_iters (int): Maximum number of iterations
        random_state (int): Random seed for reproducibility
        cluster_centers_ (np.ndarray): Coordinates of cluster centers
        labels_ (np.ndarray): Labels of each point
        inertia_ (float): Sum of squared distances of samples to their closest cluster center
    """

    def __init__(self, n_clusters: int = 3, init: str = 'random', max_iters: int = 100, 
                 tol: float = 1e-4, random_state: Optional[int] = None):
        """
        Initialize the K-Means clustering algorithm.
        
        Args:
            n_clusters: Number of clusters (default: 3)
            init: Method for initialization ('random' or 'k-means++') (default: 'random')
            max_iters: Maximum number of iterations (default: 100)
            tol: Tolerance for convergence (default: 1e-4)
            random_state: Random seed for reproducibility (default: None)
        """
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if init not in ['random', 'k-means++']:
            raise ValueError("init must be either 'random' or 'k-means++'")
        if max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")
            
        self.n_clusters = n_clusters
        self.init = init
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _euclidean_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance between points in X and Y.
        
        Args:
            X: Array of shape (n_samples_X, n_features)
            Y: Array of shape (n_samples_Y, n_features)
            
        Returns:
            Distance matrix of shape (n_samples_X, n_samples_Y)
        """
        # Broadcasting to compute pairwise distances efficiently
        X_expanded = X[:, np.newaxis, :]  # Shape: (n_samples_X, 1, n_features)
        Y_expanded = Y[np.newaxis, :, :]  # Shape: (1, n_samples_Y, n_features)
        distances = np.sqrt(np.sum((X_expanded - Y_expanded) ** 2, axis=2))
        return distances

    def _initialize_centers_random(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize cluster centers randomly.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            Initial cluster centers of shape (n_clusters, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        centers = np.zeros((self.n_clusters, n_features))
        
        for i in range(self.n_clusters):
            idx = np.random.randint(0, n_samples)
            centers[i] = X[idx]
            
        return centers

    def _initialize_centers_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize cluster centers using k-means++ algorithm.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            Initial cluster centers of shape (n_clusters, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        centers = np.zeros((self.n_clusters, n_features))
        
        # Choose first center randomly
        centers[0] = X[np.random.randint(n_samples)]
        
        # Choose remaining centers
        for c_id in range(1, self.n_clusters):
            # Calculate distances to nearest center for each point
            distances = np.array([min([np.inner(c-x, c-x) for c in centers[:c_id]]) for x in X])
            # Convert to probabilities
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()
            
            # Select next center
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centers[c_id] = X[j]
                    break
                    
        return centers

    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Fit the K-Means clustering model to training data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            Self (for method chaining)
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        if X.shape[0] < self.n_clusters:
            raise ValueError(f"Number of samples ({X.shape[0]}) must be at least equal to n_clusters ({self.n_clusters})")

        # Initialize cluster centers
        if self.init == 'random':
            self.cluster_centers_ = self._initialize_centers_random(X)
        elif self.init == 'k-means++':
            self.cluster_centers_ = self._initialize_centers_plus_plus(X)

        prev_centers = np.copy(self.cluster_centers_)
        
        for _ in range(self.max_iters):
            # Assign points to closest cluster
            distances = self._euclidean_distance(X, self.cluster_centers_)
            self.labels_ = np.argmin(distances, axis=1)
            
            # Update cluster centers
            for i in range(self.n_clusters):
                mask = self.labels_ == i
                if np.any(mask):
                    self.cluster_centers_[i] = X[mask].mean(axis=0)
                else:
                    # If a cluster has no points assigned, keep the old center
                    pass
            
            # Check for convergence
            if np.allclose(prev_centers, self.cluster_centers_, atol=self.tol):
                break
                
            prev_centers = np.copy(self.cluster_centers_)
        
        # Calculate inertia (sum of squared distances to closest cluster center)
        distances = self._euclidean_distance(X, self.cluster_centers_)
        self.inertia_ = np.sum([distances[i, label]**2 for i, label in enumerate(self.labels_)])
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster for each sample in X.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Cluster labels of shape (n_samples,)
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if X.ndim != 2 or X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(f"X must have {self.cluster_centers_.shape[1]} features, got {X.shape[1]}")
        
        distances = self._euclidean_distance(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and predict cluster labels for training data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            Cluster labels of shape (n_samples,)
        """
        return self.fit(X).labels_