"""
Example demonstrating how to use the Logistic Regression implementation from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.ml_from_scratch.models import LogisticRegression
from src.ml_from_scratch.datasets import make_classification
from src.ml_from_scratch.datasets import train_test_split
from src.ml_from_scratch.utils import plot_decision_boundary


def main():
    # Generate a synthetic dataset
    print("Generating synthetic dataset...")
    X, y = make_classification(n_samples=300, n_features=2, n_classes=2, random_state=42)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create and train the model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(fit_intercept=True)
    model.fit(X_train, y_train, lr=0.01, epochs=1000)
    
    # Make predictions
    print("Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nModel Performance:")
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    
    # Visualize the decision boundary
    plot_decision_boundary(model, X_test, y_test, title='Logistic Regression Decision Boundary')
    
    print(f"\nModel Parameters:")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")


if __name__ == "__main__":
    main()