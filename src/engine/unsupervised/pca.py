"""
Principal Component Analysis (PCA) implementation for the ML From Scratch library.

Contains the PCA class implementing dimensionality reduction using PCA.
"""

import numpy as np
from typing import Optional
from ..math.linear_algebra import power_iteration


class PCA:
    """
    Principal Component Analysis for dimensionality reduction.
    
    Attributes:
        n_components (int): Number of principal components to keep
        components_ (np.ndarray): Principal axes in feature space
        explained_variance_ (np.ndarray): Variance explained by each component
        explained_variance_ratio_ (np.ndarray): Proportion of variance explained by each component
        cum_explained_variance_ratio_ (np.ndarray): Cumulative proportion of variance explained
        mean_ (np.ndarray): Mean of the training data
    """

    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize the PCA model.
        
        Args:
            n_components: Number of components to keep (if None, keep all components)
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.cum_explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit the PCA model to training data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            Self (for method chaining)
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

        n_samples, n_features = X.shape
        
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)
        elif self.n_components > min(n_samples, n_features):
            raise ValueError(f"n_components must be <= min(n_samples, n_features) = {min(n_samples, n_features)}")
        elif self.n_components <= 0:
            raise ValueError("n_components must be positive")

        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrix
        # Using (n_samples - 1) for unbiased estimator
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # Compute eigenvalues and eigenvectors
        # We'll use numpy's built-in function for efficiency and numerical stability
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Keep only the top n_components
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        self.cum_explained_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data to the principal component space.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if X.ndim != 2 or X.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"X must have {self.mean_.shape[0]} features, got {X.shape[1]}")

        # Center the data using the training mean
        X_centered = X - self.mean_
        
        # Project onto principal components
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the PCA model and transform the data in one step.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.
        
        Args:
            X_transformed: Data in principal component space of shape (n_samples, n_components)
            
        Returns:
            Data in original space of shape (n_samples, n_original_features)
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if X_transformed.ndim != 2 or X_transformed.shape[1] != self.n_components:
            raise ValueError(f"X_transformed must have {self.n_components} features, got {X_transformed.shape[1]}")

        # Project back to original space
        X_reconstructed = X_transformed @ self.components_
        
        # Add back the mean
        return X_reconstructed + self.mean_