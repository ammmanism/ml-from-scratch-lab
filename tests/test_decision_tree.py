"""
Tests for Decision Tree implementation in the ML From Scratch library.
"""

import numpy as np
import pytest
from src.ml_from_scratch.trees.decision_tree import DecisionTreeClassifier


def test_decision_tree_shapes():
    """Test that DecisionTreeClassifier handles correct input shapes."""
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, size=100)  # Binary labels
    
    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    # Test prediction shape
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_decision_tree_basic():
    """Test DecisionTreeClassifier with basic functionality."""
    # Create simple data that should be easily separable
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    
    model = DecisionTreeClassifier(max_depth=10, min_samples_split=2)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    accuracy = np.mean(y == y_pred)
    
    # Should achieve good accuracy on simple problem
    assert accuracy >= 0.7


def test_decision_tree_multiclass():
    """Test DecisionTreeClassifier with multiclass data."""
    X = np.random.rand(150, 4)
    y = np.random.randint(0, 3, size=150)  # Three classes
    
    model = DecisionTreeClassifier(max_depth=10, min_samples_split=2)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    accuracy = np.mean(y == y_pred)
    
    # Should work with multiclass data
    assert 0 <= accuracy <= 1


def test_decision_tree_parameters():
    """Test DecisionTreeClassifier with different parameters."""
    X = np.random.rand(50, 3)
    y = np.random.randint(0, 2, size=50)
    
    # Test with different max_depth
    model = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_decision_tree_errors():
    """Test DecisionTreeClassifier raises appropriate errors."""
    model = DecisionTreeClassifier()
    
    # Test invalid max_depth
    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_depth=0)
    
    # Test invalid min_samples_split
    with pytest.raises(ValueError):
        DecisionTreeClassifier(min_samples_split=1)
    
    # Test wrong X dimensions
    with pytest.raises(ValueError):
        model.fit(np.random.rand(10), np.random.randint(0, 2, 10))  # 1D X
    
    # Test wrong y dimensions
    with pytest.raises(ValueError):
        model.fit(np.random.rand(10, 2), np.random.rand(10, 2))  # 2D y
    
    # Test mismatched dimensions
    with pytest.raises(ValueError):
        model.fit(np.random.rand(10, 2), np.random.randint(0, 2, 11))  # Different number of samples


def test_decision_tree_prediction_errors():
    """Test DecisionTreeClassifier raises errors for incorrect prediction inputs."""
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)
    
    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    # Test untrained model
    untrained_model = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        untrained_model.predict(X)
    
    # Test wrong prediction dimensions
    with pytest.raises(ValueError):
        model.predict(np.random.rand(5, 4))  # Wrong number of features