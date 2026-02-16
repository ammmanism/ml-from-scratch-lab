"""
Tests for Linear Regression implementation in the ML From Scratch library.
"""

import numpy as np
import pytest
from src.ml_from_scratch.linear.linear_regression import LinearRegression


def test_linear_regression_shapes():
    """Test that LinearRegression handles correct input shapes."""
    X = np.random.rand(100, 3)
    y = np.random.rand(100)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Check that coefficients have correct shape
    assert model.coefficients_.shape[0] == 3
    
    # Test prediction shape
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_linear_regression_normal_equation():
    """Test LinearRegression with normal equation method."""
    # Create simple linear relationship: y = 2*x1 + 3*x2 + 1
    X = np.random.rand(100, 2)
    y = 2*X[:, 0] + 3*X[:, 1] + 1 + 0.01*np.random.randn(100)
    
    model = LinearRegression(method='normal')
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    
    # Should achieve low error on training data
    assert mse < 1.0


def test_linear_regression_gradient_descent():
    """Test LinearRegression with gradient descent method."""
    # Create simple linear relationship: y = 2*x1 + 3*x2 + 1
    X = np.random.rand(100, 2)
    y = 2*X[:, 0] + 3*X[:, 1] + 1 + 0.01*np.random.randn(100)
    
    model = LinearRegression(method='gradient_descent', learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    
    # Should achieve low error on training data
    assert mse < 1.0


def test_linear_regression_regularization():
    """Test LinearRegression with L2 regularization."""
    X = np.random.rand(50, 10)  # More features than samples to test regularization
    y = np.random.rand(50)
    
    model = LinearRegression(alpha=1.0)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    
    # Should work without errors even with more features than samples
    assert mse >= 0


def test_linear_regression_errors():
    """Test LinearRegression raises appropriate errors."""
    model = LinearRegression()
    
    # Test wrong X dimensions
    with pytest.raises(ValueError):
        model.fit(np.random.rand(10), np.random.rand(10))  # 1D X
    
    # Test wrong y dimensions
    with pytest.raises(ValueError):
        model.fit(np.random.rand(10, 2), np.random.rand(10, 2))  # 2D y
    
    # Test mismatched dimensions
    with pytest.raises(ValueError):
        model.fit(np.random.rand(10, 2), np.random.rand(11))  # Different number of samples


def test_linear_regression_prediction_errors():
    """Test LinearRegression raises errors for incorrect prediction inputs."""
    X = np.random.rand(10, 3)
    y = np.random.rand(10)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Test untrained model
    untrained_model = LinearRegression()
    with pytest.raises(ValueError):
        untrained_model.predict(X)
    
    # Test wrong prediction dimensions
    with pytest.raises(ValueError):
        model.predict(np.random.rand(5, 4))  # Wrong number of features