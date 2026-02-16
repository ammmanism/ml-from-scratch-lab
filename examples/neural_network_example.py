"""
Example demonstrating how to use the Neural Network implementation from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.ml_from_scratch.neural import Sequential, Dense
from src.ml_from_scratch.neural.activations import ReLU, Sigmoid
from src.ml_from_scratch.optimizers import Adam
from src.ml_from_scratch.core.losses import BinaryCrossEntropy
from src.ml_from_scratch.datasets import make_classification
from src.ml_from_scratch.datasets import train_test_split


def main():
    # Generate a synthetic dataset
    print("Generating synthetic dataset...")
    X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, random_state=42)
    
    # Reshape y to be compatible with our neural network
    y = y.reshape(-1, 1)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create a neural network
    print("\nCreating neural network...")
    model = Sequential()
    model.add(Dense(units=10, input_dim=2, activation=ReLU()))
    model.add(Dense(units=5, activation=ReLU()))
    model.add(Dense(units=1, activation=Sigmoid()))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss=BinaryCrossEntropy())
    
    # Train the model
    print("Training the neural network...")
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=True)
    
    # Make predictions
    print("Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Convert probabilities to binary predictions
    y_pred_train_binary = (y_pred_train >= 0.5).astype(int)
    y_pred_test_binary = (y_pred_test >= 0.5).astype(int)
    
    # Calculate accuracy
    train_accuracy = np.mean(y_train == y_pred_train_binary)
    test_accuracy = np.mean(y_test == y_pred_test_binary)
    
    print(f"\nModel Performance:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Visualize some results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Show the dataset
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.title('Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    
    # Plot 2: Show predictions on test set
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test_binary.ravel(), 
                         cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.title('Neural Network Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()