# Mathematical Foundations 

This document serves as a cheat sheet for the primary derivations implemented across the codebase. For deep dives, examine the Jupyter notebooks in `notebooks/`.

## 1. Linear Regression
**Objective**: Minimize Mean Squared Error (MSE).
$$ J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)})^2 $$

### Normal Equation (Closed-form solution)
$$ \\theta = (X^T X)^{-1} X^T y $$

### Gradient Descent
$$ \\frac{\\partial J}{\\partial \\theta} = \\frac{1}{m} X^T (X\\theta - y) $$
$$ \\theta_{new} = \\theta_{old} - \\alpha \\frac{\\partial J}{\\partial \\theta} $$

## 2. Logistic Regression
**Objective**: Minimize Binary Cross-Entropy Loss (Log-Loss).
$$ J(\\theta) = -\\frac{1}{m} \\sum_{i=1}^{m} [y^{(i)}\\log(h_\\theta(x^{(i)})) + (1-y^{(i)})\\log(1-h_\\theta(x^{(i)}))] $$

**Sigmoid Function**:
$$ h_\\theta(x) = \\sigma(z) = \\frac{1}{1 + e^{-z}} $$

**Gradient**:
$$ \\frac{\\partial J}{\\partial \\theta} = \\frac{1}{m} X^T (\\sigma(X\\theta) - y) $$

## 3. Backpropagation (Multi-Layer Perceptron)
The chain rule is used to compute gradients backward from the output layer to the weights.

Let $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$ be the linear step for layer $l$.
Let $A^{[l]} = g(Z^{[l]})$ be the activation.

**Output Layer Error**:
$$ dZ^{[L]} = A^{[L]} - Y $$

**Hidden Layer Error**:
$$ dZ^{[l]} = (W^{[l+1]})^T dZ^{[l+1]} \\circ g'(Z^{[l]}) $$

**Gradients**:
$$ dW^{[l]} = \\frac{1}{m} dZ^{[l]} (A^{[l-1]})^T $$
$$ db^{[l]} = \\frac{1}{m} \\sum dZ^{[l]} $$
