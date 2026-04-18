import numpy as np
from typing import Tuple, Callable

class GaussianProcessRegressor:
    """Gaussian Process Regression using an RBF (Radial Basis Function) Kernel.
    
    Attributes:
        kernel: The covariance function.
        sigma_y: Noise variance on the observations.
    """
    def __init__(self, length_scale: float = 1.0, sigma_f: float = 1.0, sigma_y: float = 1e-8):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_y = sigma_y
        self.X_train = None
        self.y_train = None
        self.K = None
        self.K_inv = None

    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Computes the Squared Exponential (RBF) kernel matrix."""
        # Using broadcasting to compute squared Euclidean distance
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.length_scale**2 * sqdist)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the Gaussian Process to the data.
        
        In GP, "fitting" is mostly just computing the covariance matrix and its inverse.
        """
        self.X_train = X
        self.y_train = y
        self.K = self.rbf_kernel(X, X) + self.sigma_y**2 * np.eye(len(X))
        
        # We use Cholesky decomposition for numerically stable inversion
        try:
            L = np.linalg.cholesky(self.K)
            self.K_inv = np.linalg.inv(L.T) @ np.linalg.inv(L)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is not positive definite
            self.K_inv = np.linalg.pinv(self.K)

    def predict(self, X_test: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts the mean and standard deviation at new points."""
        K_s = self.rbf_kernel(self.X_train, X_test)
        K_ss = self.rbf_kernel(X_test, X_test) + 1e-8 * np.eye(len(X_test))

        # Mean prediction
        mu_s = K_s.T @ self.K_inv @ self.y_train

        # Covariance prediction
        cov_s = K_ss - K_s.T @ self.K_inv @ K_s
        
        if return_std:
            std_s = np.sqrt(np.diag(cov_s))
            return mu_s, std_s
        return mu_s
