"""
Tests for MLP implementation in the ML From Scratch library.
"""

import numpy as np
import pytest
from src.ml_from_scratch.neural.mlp import MLPClassifier


def test_mlp_shapes():
    """Test that MLPClassifier handles correct input shapes."""
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, size=100)  # Binary labels
    
    model = MLPClassifier(hidden_layers=[10])
    model.fit(X, y)
    
    # Test prediction shape
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    
    # Test probability shape
    y_proba = model.predict_proba(X)
    assert y_proba.shape == (y.shape[0], 2)  # 2 classes


def test_mlp_basic():
    """Test MLPClassifier with basic functionality."""
    # Create simple linearly separable data
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    
    model = MLPClassifier(hidden_layers=[10], learning_rate=0.1, n_iterations=100)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    accuracy = np.mean(y == y_pred)
    
    # Should achieve reasonable accuracy on simple problem
    assert accuracy >= 0.6


def test_mlp_multiclass():
    """Test MLPClassifier with multiclass data."""
    X = np.random.rand(150, 4)
    y = np.random.randint(0, 3, size=150)  # Three classes
    
    model = MLPClassifier(hidden_layers=[20, 10], learning_rate=0.01, n_iterations=100)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    accuracy = np.mean(y == y_pred)
    
    # Should work with multiclass data
    assert 0 <= accuracy <= 1


def test_mlp_different_architectures():
    """Test MLPClassifier with different architectures."""
    X = np.random.rand(50, 3)
    y = np.random.randint(0, 2, size=50)
    
    # Test with different hidden layer configurations
    architectures = [[5], [10, 5], [20, 10, 5]]
    
    for arch in architectures:
        model = MLPClassifier(hidden_layers=arch, learning_rate=0.01, n_iterations=50)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape
        
        y_proba = model.predict_proba(X)
        assert y_proba.shape[0] == y.shape[0]


def test_mlp_regularization():
    """Test MLPClassifier with regularization."""
    X = np.random.rand(50, 10)  # More features to test regularization
    y = np.random.randint(0, 2, size=50)
    
    model = MLPClassifier(hidden_layers=[15], learning_rate=0.01, n_iterations=50, alpha=0.1)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    accuracy = np.mean(y == y_pred)
    
    # Should work without errors
    assert 0 <= accuracy <= 1


def test_mlp_errors():
    """Test MLPClassifier raises appropriate errors."""
    model = MLPClassifier()
    
    # Test invalid learning rate
    with pytest.raises(ValueError):
        MLPClassifier(learning_rate=0)
    
    # Test invalid n_iterations
    with pytest.raises(ValueError):
        MLPClassifier(n_iterations=0)
    
    # Test invalid alpha
    with pytest.raises(ValueError):
        MLPClassifier(alpha=-1)
    
    # Test wrong X dimensions
    with pytest.raises(ValueError):
        model.fit(np.random.rand(10), np.random.randint(0, 2, 10))  # 1D X
    
    # Test wrong y dimensions
    with pytest.raises(ValueError):
        model.fit(np.random.rand(10, 2), np.random.rand(10, 2))  # 2D y
    
    # Test mismatched dimensions
    with pytest.raises(ValueError):
        model.fit(np.random.rand(10, 2), np.random.randint(0, 2, 11))  # Different number of samples


def test_mlp_prediction_errors():
    """Test MLPClassifier raises errors for incorrect prediction inputs."""
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)
    
    model = MLPClassifier(hidden_layers=[5])
    model.fit(X, y)
    
    # Test untrained model
    untrained_model = MLPClassifier()
    with pytest.raises(ValueError):
        untrained_model.predict(X)
    
    # Test untrained model for probabilities
    with pytest.raises(ValueError):
        untrained_model.predict_proba(X)
    
    # Test wrong prediction dimensions
    with pytest.raises(ValueError):
        model.predict(np.random.rand(5, 4))  # Wrong number of features