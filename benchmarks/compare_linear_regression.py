"""
Benchmark script comparing our Linear Regression implementation with scikit-learn.
"""

import numpy as np
import time
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error
from src.ml_from_scratch.models import LinearRegression
from src.ml_from_scratch.datasets import make_regression
from src.ml_from_scratch.utils import mean_squared_error as scratch_mse


def benchmark_linear_regression():
    # Generate dataset
    print("Generating dataset...")
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    
    # Split the data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Benchmark our implementation
    print("\nBenchmarking our Linear Regression implementation...")
    start_time = time.time()
    
    scratch_model = LinearRegression(fit_intercept=True)
    scratch_model.fit(X_train, y_train, lr=0.01, epochs=1000)
    
    scratch_train_time = time.time() - start_time
    
    # Make predictions
    start_time = time.time()
    scratch_predictions = scratch_model.predict(X_test)
    scratch_predict_time = time.time() - start_time
    
    scratch_mse_score = scratch_mse(y_test, scratch_predictions)
    scratch_r2_score = scratch_model.score(X_test, y_test)
    
    print(f"Our implementation:")
    print(f"  Training time: {scratch_train_time:.4f}s")
    print(f"  Prediction time: {scratch_predict_time:.6f}s")
    print(f"  MSE: {scratch_mse_score:.6f}")
    print(f"  R² Score: {scratch_r2_score:.6f}")
    
    # Benchmark scikit-learn implementation
    print("\nBenchmarking scikit-learn Linear Regression...")
    start_time = time.time()
    
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X_train, y_train)
    
    sklearn_train_time = time.time() - start_time
    
    # Make predictions
    start_time = time.time()
    sklearn_predictions = sklearn_model.predict(X_test)
    sklearn_predict_time = time.time() - start_time
    
    sklearn_mse_score = mean_squared_error(y_test, sklearn_predictions)
    sklearn_r2_score = sklearn_model.score(X_test, y_test)
    
    print(f"Scikit-learn implementation:")
    print(f"  Training time: {sklearn_train_time:.4f}s")
    print(f"  Prediction time: {sklearn_predict_time:.6f}s")
    print(f"  MSE: {sklearn_mse_score:.6f}")
    print(f"  R² Score: {sklearn_r2_score:.6f}")
    
    # Compare results
    print("\nComparison:")
    print(f"  MSE difference: {abs(scratch_mse_score - sklearn_mse_score):.8f}")
    print(f"  R² difference: {abs(scratch_r2_score - sklearn_r2_score):.8f}")
    print(f"  Training time ratio (Scratch/Scikit-learn): {scratch_train_time / sklearn_train_time:.2f}x")
    print(f"  Prediction time ratio (Scratch/Scikit-learn): {scratch_predict_time / sklearn_predict_time:.2f}x")


if __name__ == "__main__":
    benchmark_linear_regression()