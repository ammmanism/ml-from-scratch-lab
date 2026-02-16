"""
Example demonstrating how to use the Linear Regression implementation from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.ml_from_scratch.models import LinearRegression
from src.ml_from_scratch.datasets import make_regression
from src.ml_from_scratch.datasets import train_test_split


def main():
    # Generate a synthetic dataset
    print("Generating synthetic dataset...")
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create and train the model
    print("\nTraining Linear Regression model...")
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train, lr=0.01, epochs=1000)
    
    # Make predictions
    print("Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nModel Performance:")
    print(f"Training R² Score: {train_score:.4f}")
    print(f"Test R² Score: {test_score:.4f}")
    
    # Visualize the results
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training Data')
    
    # Plot test data
    plt.scatter(X_test, y_test, color='red', alpha=0.6, label='Test Data')
    
    # Plot regression line for the entire range
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range_pred = model.predict(X_range)
    plt.plot(X_range, y_range_pred, color='black', linewidth=2, label='Fitted Line')
    
    plt.title('Linear Regression Example')
    plt.xlabel('Feature Value')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"\nModel Parameters:")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")


if __name__ == "__main__":
    main()