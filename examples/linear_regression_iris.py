"""
Example script demonstrating Linear Regression on the Iris dataset.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from src.ml_from_scratch.linear.linear_regression import LinearRegression
from src.ml_from_scratch.utils import train_test_split


def main():
    # Load the iris dataset
    iris = load_iris()
    X = iris.data  # Features: sepal length, sepal width, petal length, petal width
    y = iris.target  # Target: species (0=setosa, 1=versicolor, 2=virginica)
    
    # For linear regression, let's predict petal width from other features
    X = np.delete(X, 3, axis=1)  # Remove petal width from features
    y = iris.data[:, 3]  # Use petal width as target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the linear regression model
    model = LinearRegression(method='normal')
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate mean squared error
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    print("Linear Regression on Iris Dataset")
    print("=================================")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Testing MSE: {test_mse:.4f}")
    print(f"Coefficients: {model.coefficients_}")
    print(f"Intercept: {model.intercept_:.4f}")


if __name__ == "__main__":
    main()