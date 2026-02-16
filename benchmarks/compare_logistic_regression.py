"""
Benchmark script comparing our Logistic Regression implementation with scikit-learn.
"""

import numpy as np
import time
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from src.ml_from_scratch.models import LogisticRegression
from src.ml_from_scratch.datasets import make_classification
from src.ml_from_scratch.utils import accuracy_score as scratch_accuracy


def benchmark_logistic_regression():
    # Generate dataset
    print("Generating dataset...")
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    
    # Split the data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Benchmark our implementation
    print("\nBenchmarking our Logistic Regression implementation...")
    start_time = time.time()
    
    scratch_model = LogisticRegression(fit_intercept=True)
    scratch_model.fit(X_train, y_train, lr=0.01, epochs=1000)
    
    scratch_train_time = time.time() - start_time
    
    # Make predictions
    start_time = time.time()
    scratch_predictions = scratch_model.predict(X_test)
    scratch_predict_time = time.time() - start_time
    
    scratch_accuracy_score = scratch_accuracy(y_test, scratch_predictions)
    
    print(f"Our implementation:")
    print(f"  Training time: {scratch_train_time:.4f}s")
    print(f"  Prediction time: {scratch_predict_time:.6f}s")
    print(f"  Accuracy: {scratch_accuracy_score:.6f}")
    
    # Benchmark scikit-learn implementation
    print("\nBenchmarking scikit-learn Logistic Regression...")
    start_time = time.time()
    
    sklearn_model = SklearnLogisticRegression(max_iter=1000)
    sklearn_model.fit(X_train, y_train)
    
    sklearn_train_time = time.time() - start_time
    
    # Make predictions
    start_time = time.time()
    sklearn_predictions = sklearn_model.predict(X_test)
    sklearn_predict_time = time.time() - start_time
    
    sklearn_accuracy_score = accuracy_score(y_test, sklearn_predictions)
    
    print(f"Scikit-learn implementation:")
    print(f"  Training time: {sklearn_train_time:.4f}s")
    print(f"  Prediction time: {sklearn_predict_time:.6f}s")
    print(f"  Accuracy: {sklearn_accuracy_score:.6f}")
    
    # Compare results
    print("\nComparison:")
    print(f"  Accuracy difference: {abs(scratch_accuracy_score - sklearn_accuracy_score):.8f}")
    print(f"  Training time ratio (Scratch/Scikit-learn): {scratch_train_time / sklearn_train_time:.2f}x")
    print(f"  Prediction time ratio (Scratch/Scikit-learn): {scratch_predict_time / sklearn_predict_time:.2f}x")


if __name__ == "__main__":
    benchmark_logistic_regression()