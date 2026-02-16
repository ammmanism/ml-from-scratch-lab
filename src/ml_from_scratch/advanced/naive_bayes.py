"""
Gaussian Naive Bayes implementation for the ML From Scratch library.

Contains the GaussianNB class implementing Gaussian Naive Bayes classifier.
"""

import numpy as np
from typing import Optional


class GaussianNB:
    """
    Gaussian Naive Bayes classifier.
    
    Attributes:
        classes (np.ndarray): Unique class labels
        class_priors (np.ndarray): Prior probabilities for each class
        mean (np.ndarray): Mean of each feature per class of shape (n_classes, n_features)
        var (np.ndarray): Variance of each feature per class of shape (n_classes, n_features)
    """

    def __init__(self):
        """Initialize the Gaussian Naive Bayes classifier."""
        self.classes = None
        self.class_priors = None
        self.mean = None
        self.var = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNB':
        """
        Fit the Gaussian Naive Bayes classifier to training data.
        
        Args:
            X: Training data features of shape (n_samples, n_features)
            y: Training data targets of shape (n_samples,) with discrete class labels
            
        Returns:
            Self (for method chaining)
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: X {X.shape[0]}, y {y.shape[0]}")

        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)
        
        # Initialize mean and variance arrays
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.class_priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            # Get samples for this class
            X_c = X[y == c]
            
            # Calculate mean and variance for each feature in this class
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)  # Use biased variance (divide by n)
            
            # Calculate class prior (probability of this class)
            self.class_priors[idx] = X_c.shape[0] / n_samples
        
        return self

    def _calculate_likelihood(self, x: np.ndarray, mean: float, var: float) -> np.ndarray:
        """
        Calculate the likelihood of a feature value given class parameters using Gaussian PDF.
        
        Args:
            x: Feature values of shape (n_features,)
            mean: Mean of the feature for a class
            var: Variance of the feature for a class
            
        Returns:
            Likelihood values
        """
        # Add small constant to variance to avoid division by zero
        eps = 1e-4
        coeff = 1.0 / np.sqrt(2.0 * np.pi * (var + eps))
        exponent = np.exp(-((x - mean) ** 2) / (2 * (var + eps)))
        return coeff * exponent

    def _calculate_class_posteriors(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the posterior probability for each class for a single sample.
        
        Args:
            x: Single sample of shape (n_features,)
            
        Returns:
            Posterior probabilities for each class
        """
        posteriors = []
        
        for idx, c in enumerate(self.classes):
            # Calculate log prior for numerical stability
            log_prior = np.log(self.class_priors[idx])
            
            # Calculate likelihood for each feature
            log_likelihood = np.sum(
                np.log(self._calculate_likelihood(x, self.mean[idx], self.var[idx]))
            )
            
            # Posterior is prior * likelihood (in log space: log prior + log likelihood)
            posterior = log_prior + log_likelihood
            posteriors.append(posterior)
        
        return np.array(posteriors)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Probabilities of shape (n_samples, n_classes) for each class
        """
        if self.classes is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if X.ndim != 2 or X.shape[1] != self.mean.shape[1]:
            raise ValueError(f"X must have {self.mean.shape[1]} features, got {X.shape[1]}")
        
        # Calculate probabilities for each sample
        probas = []
        for x in X:
            posteriors = self._calculate_class_posteriors(x)
            # Convert back from log space using softmax for numerical stability
            # First subtract max to prevent overflow
            posteriors_stable = posteriors - np.max(posteriors)
            exp_posteriors = np.exp(posteriors_stable)
            class_probas = exp_posteriors / np.sum(exp_posteriors)
            probas.append(class_probas)
        
        return np.array(probas)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted classes of shape (n_samples,) with discrete class labels
        """
        if self.classes is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if X.ndim != 2 or X.shape[1] != self.mean.shape[1]:
            raise ValueError(f"X must have {self.mean.shape[1]} features, got {X.shape[1]}")
        
        predictions = []
        for x in X:
            posteriors = self._calculate_class_posteriors(x)
            # Predict class with highest posterior probability
            class_idx = np.argmax(posteriors)
            predictions.append(self.classes[class_idx])
        
        return np.array(predictions)