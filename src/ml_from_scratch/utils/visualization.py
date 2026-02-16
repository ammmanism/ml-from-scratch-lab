"""Visualization utilities for plotting model results."""

import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Plot the decision boundary of a classifier.
    
    Args:
        model: Trained classifier with a predict method
        X: Features array of shape (n_samples, 2) - only works for 2D features
        y: Target values array of shape (n_samples,)
        title: Title for the plot
    """
    if X.shape[1] != 2:
        raise ValueError("Plotting decision boundary only works for 2D features")
    
    # Set up the mesh
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    plt.show()