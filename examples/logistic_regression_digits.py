"""
Example script demonstrating Logistic Regression on the Digits dataset.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from src.ml_from_scratch.linear.logistic_regression import LogisticRegression
from src.ml_from_scratch.utils import train_test_split


def main():
    # Load the digits dataset (binary classification: digit 0 vs digit 1)
    digits = load_digits()
    X = digits.data  # Pixel values for each image
    y = digits.target  # Digit labels
    
    # For binary classification, let's classify digit 0 vs digit 1
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the logistic regression model
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print("Logistic Regression on Digits Dataset (Binary)")
    print("============================================")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print(f"Number of features: {X.shape[1]}")


if __name__ == "__main__":
    main()