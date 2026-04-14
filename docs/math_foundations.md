# 🧬 Mathematical Foundations of ML From Scratch

> "Mathematics is the language in which God has written the universe." — Galileo Galilei

In this repository, we don't treat formulas as black boxes. We treat them as **geometric and probabilistic realities**. This document derives the bedrock principles of the library, bridging abstract theory with NumPy implementation.

---

## 🏗️ Table of Contents
1. [Linear Algebra: The Geometry of Space](#1-linear-algebra-the-geometry-of-space)
2. [Multivariate Calculus: The Sensitivity of Systems](#2-multivariate-calculus-the-sensitivity-of-systems)
3. [Probability & Information Theory](#3-probability--information-theory)
4. [Derivative Cheat Sheet](#4-derivative-cheat-sheet)

---

## 1. Linear Algebra: The Geometry of Space

### 1.1 Singular Value Decomposition (SVD)
SVD is the most profound theorem in linear algebra. Every matrix $A \in \mathbb{R}^{m \times n}$ can be seen as a transformation that rotates, stretches, and rotates again:
$$A = U \Sigma V^T$$
- **Geometry**: $V^T$ rotates the input space, $\Sigma$ stretches it along the axes, and $U$ rotates it into the output space.
- **The Eckart–Young–Mirsky Theorem**: This is the "Why" behind our PCA implementation. It states that the best rank-$k$ approximation of $A$ (in terms of Frobenius norm) is found by keeping only the top-$k$ singular values.
- **Why we use it**: Instead of computing the Covariance Matrix $(X^T X)$, which squares the condition number and leads to numerical instability, we perform SVD directly on $X$.

### 1.2 Eigenvalues and Spectral Theory
The Eigenvalue decomposition $A\mathbf{v} = \lambda \mathbf{v}$ represents the "characteristic" directions of a linear system. In our library, we use **Power Iteration** to find these directions iteratively when direct decomposition is too expensive.

---

## 2. Multivariate Calculus: The Sensitivity of Systems

### 2.1 The Jacobian Matrix
The Jacobian $\mathbf{J}$ is the "Sensitivity Matrix." If you have a function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian element $J_{ij}$ tells you exactly how much the $i$-th output changes given a tiny nudge to the $j$-th input.
- **Intuition**: In Backpropagation, the Jacobian of each layer represents the "local error sensitivity" that we chain together.

### 2.2 The Hessian and Local Curvature
The Hessian $\mathbf{H}$ is the square matrix of second-order partial derivatives.
- **Geometric Note**: The Eigenvalues of the Hessian determine the **local curvature**. Large eigenvalues mean a steep "valley," while small eigenvalues mean a "flat" region. 
- **The Problem with 2nd Order**: While Newton's Method uses $\mathbf{H}^{-1}$ to jump straight to the minimum, inverting a $10^6 \times 10^6$ Hessian is globally impossible. This is why we rely on **First-Order Momentum**.

---

## 3. Probability & Information Theory

### 3.1 Gini Impurity vs. Entropy
For our `DecisionTree` implementation, we choose between these two purity metrics:
- **Entropy**: $H(p) = -\sum p_i \log p_i$. It represents the "Expected Surprise." High entropy means total uncertainty.
- **Gini**: $G(p) = 1 - \sum p_i^2$. It is the probability of misclassifying a random element.
- **The Connection**: Gini is actually a first-order Taylor approximation of Entropy. We use Gini by default because it avoids the expensive `log` operation.

### 3.2 Maximum Likelihood Estimation (MLE)
Every model in `src/ml_from_scratch/linear` is an MLE solver. By minimizing the Negative Log-Likelihood (NLL), we are effectively finding the parameters that make the observed data "most probable."

---

## 4. Derivative Cheat Sheet
For your implementation, reference this table for analytical gradients:

| Function | $f(x)$ | $\nabla f(x)$ | Implementation Note |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1 - \sigma(x))$ | Self-referential derivative |
| **ReLU** | $\max(0, x)$ | $1$ if $x > 0$, else $0$ | Sparse activation |
| **Softmax** | $\frac{e^{x_i}}{\sum e^{x_j}}$ | $P_i(\delta_{ij} - P_j)$ | Jacobian is symmetric |
| **MSE Loss** | $\frac{1}{2}(y - \hat{y})^2$ | $(\hat{y} - y)$ | Error signal is linear |
| **Cross-Entropy** | $-y \log(\hat{y})$ | $\hat{y} - y$ | Combined with Softmax, yields a simple linear signal |

---

## 💡 10 Pitfalls When Implementing from Scratch

1. **Broadcasting Bugs**: `(n, ) + (n, 1)` results in an `(n, n)` matrix in NumPy. Always use `.reshape(-1, 1)`.
2. **Softmax Overflow**: Always subtract `np.max(x)` before exponentiating.
3. **Internal Division**: Forgetting to divide the gradient by the batch size $m$ leads to unstable learning rates.
4. **Incorrect Initialization**: Initializing all weights to zero makes all neurons in a layer compute the same thing. Use **He** or **Xavier**.
5. **Numerical Underflow**: Using `log(y)` where $y$ is very small. Always use `log(y + eps)`.
6. **In-place Mutation**: Modifying a matrix during the forward pass that you need for the backward pass.
7. **Gradient Clipping**: Ignoring Exploding Gradients in deep networks.
8. **Shuffling**: Not shuffling your data before SGD leads to "Epoch Bias."
9. **Regularization Leakage**: Applying L2 penalty to the bias term $\theta_0$ (usually unnecessary).
10. **Chain Rule Order**: Multiplying Jacobians in the wrong order ($W \cdot \delta$ vs $\delta \cdot W$).

---

### **Further Reading**
- *Pattern Recognition and Machine Learning* (Bishop, 2006)
- *Deep Learning* (Goodfellow, Bengio, Courville, 2016)
- *Attention is All You Need* (Vaswani et al., 2017)
