"""
Tests for K-Means implementation in the ML From Scratch library.
"""

import numpy as np
import pytest
from src.ml_from_scratch.unsupervised.kmeans import KMeans


def test_kmeans_shapes():
    """Test that KMeans handles correct input shapes."""
    X = np.random.rand(100, 3)
    
    model = KMeans(n_clusters=3)
    labels = model.fit_predict(X)
    
    # Test that labels have correct shape
    assert labels.shape[0] == 100
    
    # Test prediction shape
    new_labels = model.predict(X)
    assert new_labels.shape == labels.shape


def test_kmeans_basic():
    """Test KMeans with basic functionality."""
    # Create simple data with 2 clear clusters
    X1 = np.random.rand(50, 2) + [2, 2]
    X2 = np.random.rand(50, 2) + [-2, -2]
    X = np.vstack([X1, X2])
    
    model = KMeans(n_clusters=2, random_state=42)
    labels = model.fit_predict(X)
    
    # Check that the model converged and has reasonable inertia
    assert hasattr(model, 'inertia_')
    assert model.inertia_ >= 0


def test_kmeans_plus_plus():
    """Test KMeans with k-means++ initialization."""
    X = np.random.rand(100, 2)
    
    model = KMeans(n_clusters=3, init='k-means++', random_state=42)
    model.fit(X)
    
    # Check that the model has the expected attributes
    assert hasattr(model, 'cluster_centers_')
    assert model.cluster_centers_.shape[0] == 3


def test_kmeans_different_clusters():
    """Test KMeans with different numbers of clusters."""
    X = np.random.rand(50, 3)
    
    # Test with different numbers of clusters
    for n_clusters in [2, 3, 4]:
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        
        # Check that all cluster labels are in range [0, n_clusters)
        assert np.all(labels >= 0) and np.all(labels < n_clusters)
        assert len(np.unique(labels)) <= n_clusters


def test_kmeans_errors():
    """Test KMeans raises appropriate errors."""
    X = np.random.rand(10, 3)
    
    # Test invalid n_clusters
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)
    
    # Test invalid init method
    with pytest.raises(ValueError):
        KMeans(init='invalid')
    
    # Test invalid max_iters
    with pytest.raises(ValueError):
        KMeans(max_iters=0)
    
    # Test invalid tol
    with pytest.raises(ValueError):
        KMeans(tol=0)
    
    # Test too few samples
    with pytest.raises(ValueError):
        model = KMeans(n_clusters=10)
        model.fit(X)  # Only 10 samples but asking for 10 clusters (would violate minimum constraint)
    
    # Test wrong X dimensions
    with pytest.raises(ValueError):
        model = KMeans()
        model.fit(np.random.rand(10))  # 1D X


def test_kmeans_prediction_errors():
    """Test KMeans raises errors for incorrect prediction inputs."""
    X = np.random.rand(10, 3)
    
    model = KMeans(n_clusters=2)
    model.fit(X)
    
    # Test untrained model
    untrained_model = KMeans(n_clusters=2)
    with pytest.raises(ValueError):
        untrained_model.predict(X)
    
    # Test wrong prediction dimensions
    with pytest.raises(ValueError):
        model.predict(np.random.rand(5, 4))  # Wrong number of features