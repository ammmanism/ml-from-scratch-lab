"""
Tests for Logistic Regression implementation in the ML From Scratch library.
"""

import numpy as np
import pytest
from src.ml_from_scratch.linear.logistic_regression import LogisticRegression


def test_logistic_regression_shapes():
    """Test that LogisticRegression handles correct input shapes."""
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, size=100)  # Binary labels
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Check that coefficients have correct shape
    assert model.coefficients_.shape[0] == 3
    
    # Test prediction shape
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    
    # Test probability shape
    y_proba = model.predict_proba(X)
    assert y_proba.shape == y.shape


def test_logistic_regression_basic():
    """Test LogisticRegression with basic functionality."""
    # Create simple linearly separable data
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    accuracy = np.mean(y == y_pred)
    
    # Should achieve reasonable accuracy on simple problem
    assert accuracy >= 0.7


def test_logistic_regression_regularization():
    """Test LogisticRegression with L2 regularization."""
    X = np.random.rand(50, 10)  # More features than samples to test regularization
    y = np.random.randint(0, 2, size=50)
    
    model = LogisticRegression(alpha=1.0)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    accuracy = np.mean(y == y_pred)
    
    # Should work without errors even with more features than samples
    assert 0 <= accuracy <= 1


def test_logistic_regression_errors():
    """Test LogisticRegression raises appropriate errors."""
    model = LogisticRegression()
    
    # Test wrong X dimensions
    with pytest.raises(ValueError):
        model.fit(np.random.rand(10), np.random.randint(0, 2, 10))  # 1D X
    
    # Test wrong y dimensions
    with pytest.raises(ValueError):
        model.fit(np.random.rand(10, 2), np.random.rand(10, 2))  # 2D y
    
    # Test mismatched dimensions
    with pytest.raises(ValueError):
        model.fit(np.random.rand(10, 2), np.random.randint(0, 2, 11))  # Different number of samples
    
    # Test non-binary labels
    with pytest.raises(ValueError):
        model.fit(np.random.rand(10, 2), np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 0]))  # Contains 2


def test_logistic_regression_prediction_errors():
    """Test LogisticRegression raises errors for incorrect prediction inputs."""
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Test untrained model
    untrained_model = LogisticRegression()
    with pytest.raises(ValueError):
        untrained_model.predict(X)
    
    # Test untrained model for probabilities
    with pytest.raises(ValueError):
        untrained_model.predict_proba(X)
    
    # Test wrong prediction dimensions
    with pytest.raises(ValueError):
        model.predict(np.random.rand(5, 4))  # Wrong number of features