# Mathematical Foundations

> *Every formula in a paper should map to one line of code. This document ensures you can write that line.*

---

## Table of Contents

- [Linear Algebra](#1-linear-algebra)
  - [Vector Norms](#11-vector-norms)
  - [Matrix Decompositions](#12-matrix-decompositions)
  - [SVD — The Crown Jewel](#13-svd--the-crown-jewel)
- [Calculus & Optimization](#2-calculus--optimization)
  - [Gradients and Jacobians](#21-gradients-and-jacobians)
  - [Chain Rule & Computational Graphs](#22-chain-rule--computational-graphs)
  - [Hessians and Second-Order Methods](#23-hessians-and-second-order-methods)
- [Probability & Information Theory](#3-probability--information-theory)
  - [Maximum Likelihood Estimation](#31-maximum-likelihood-estimation)
  - [KL Divergence & Cross-Entropy](#32-kl-divergence--cross-entropy)
  - [Information Gain & Entropy](#33-information-gain--entropy)
- [Quick Reference: Derivatives](#4-quick-reference-derivatives)
- [Numerical Stability Notes](#5-numerical-stability-notes)
- [Further Reading](#6-further-reading)

---

## 1. Linear Algebra

### 1.1 Vector Norms

A norm $\|\cdot\|$ measures "length" in a vector space. Three norms dominate ML:

**L1 norm (Manhattan):**
$$\|x\|_1 = \sum_{i=1}^n |x_i|$$

Promotes **sparsity**. The diamond-shaped unit ball means the optimum of an L1-penalized objective often lands exactly at a coordinate axis (one weight is exactly zero). This is why Lasso produces sparse models.

**L2 norm (Euclidean):**
$$\|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$$

Smooth everywhere. Its gradient $\nabla \|x\|_2^2 = 2x$ makes it easy to optimize — which is why Ridge regression has an analytical solution.

**L∞ norm:**
$$\|x\|_\infty = \max_i |x_i|$$

The maximum absolute component. Used in robustness analysis.

```python
import numpy as np

x = np.array([3.0, -4.0, 0.0])
l1 = np.sum(np.abs(x))          # 7.0
l2 = np.linalg.norm(x)          # 5.0
linf = np.max(np.abs(x))        # 4.0
```

---

### 1.2 Matrix Decompositions

**Eigendecomposition:** For a square matrix $A \in \mathbb{R}^{n \times n}$:
$$A = Q \Lambda Q^{-1}$$

where $Q$ columns are eigenvectors, $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$.

**Geometric intuition:** Eigenvectors are the directions that $A$ doesn't rotate — it only stretches them by their eigenvalue. When $\lambda < 0$, it flips direction; when $|\lambda| < 1$, it contracts.

For **symmetric** $A = A^\top$ (covariance matrices, Gram matrices), the decomposition is orthogonal:
$$A = Q \Lambda Q^\top, \quad Q^\top Q = I$$

This is the **spectral theorem** — symmetric matrices always have real eigenvalues and orthonormal eigenvectors.

```python
A = np.array([[4, 2], [2, 3]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eigh(A)  # eigh for symmetric
# eigenvalues: [1.76, 5.24]  (real, ordered ascending)
```

---

### 1.3 SVD — The Crown Jewel

For any matrix $A \in \mathbb{R}^{m \times n}$:

$$A = U \Sigma V^\top$$

where:
- $U \in \mathbb{R}^{m \times m}$: orthonormal basis for **column space** of $A$
- $V \in \mathbb{R}^{n \times n}$: orthonormal basis for **row space** of $A$
- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal with singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$

**Physical intuition:** SVD says "any linear map is just a rotation ($V^\top$), then axis-aligned scaling ($\Sigma$), then another rotation ($U$)." The singular values tell you *how much* scaling happens in each direction.

**The Eckart–Young Theorem** (best low-rank approximation):
$$A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top = \underset{\text{rank}(B) \leq k}{\arg\min} \|A - B\|_F$$

This is the mathematical foundation of PCA: the first $k$ principal components capture the maximum variance in $k$ dimensions.

```python
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Rank-k approximation
k = 2
A_k = (U[:, :k] * S[:k]) @ Vt[:k, :]

# PCA via SVD (on centered data)
X_centered = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
# Principal components = rows of Vt
# Projections = X_centered @ Vt[:k].T
# Explained variance ratio = S[:k]**2 / (S**2).sum()
```

> **Connection to eigendecomposition:** The singular values of $A$ are the square roots of the eigenvalues of $A^\top A$. The columns of $V$ are its eigenvectors.

---

## 2. Calculus & Optimization

### 2.1 Gradients and Jacobians

For $f: \mathbb{R}^n \to \mathbb{R}$, the **gradient** is the vector of partial derivatives:
$$\nabla f(x) = \left[\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right]^\top$$

**Key property:** $\nabla f$ always points in the direction of **steepest ascent**. Gradient descent subtracts it: $x \leftarrow x - \alpha \nabla f(x)$.

For $f: \mathbb{R}^n \to \mathbb{R}^m$ (vector-valued), the **Jacobian** $J \in \mathbb{R}^{m \times n}$:
$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

**Sensitivity interpretation:** $J_{ij}$ answers "if I perturb input $x_j$ by $\epsilon$, how much does output $f_i$ change?" The Jacobian is the "sensitivity matrix" of a transformation.

```python
def numerical_jacobian(f, x, eps=1e-5):
    """Finite-difference Jacobian. Use to verify analytical gradients."""
    fx = f(x)
    m, n = len(fx), len(x)
    J = np.zeros((m, n))
    for j in range(n):
        x_plus = x.copy(); x_plus[j] += eps
        x_minus = x.copy(); x_minus[j] -= eps
        J[:, j] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return J
```

---

### 2.2 Chain Rule & Computational Graphs

For composed functions $L = f(g(x))$:
$$\frac{\partial L}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}$$

In neural networks, every layer is a node in the computational graph. Backpropagation is **reverse-mode automatic differentiation** — it propagates $\frac{\partial L}{\partial \text{output}}$ backward through each node using the chain rule.

**Example: Dense layer** $z = Wx + b$, then $a = \sigma(z)$:

Forward:
$$z = Wx + b, \quad a = \sigma(z)$$

Backward (given upstream gradient $\delta = \frac{\partial L}{\partial a}$):
$$\frac{\partial L}{\partial z} = \delta \odot \sigma'(z)$$
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial z} \cdot x^\top$$
$$\frac{\partial L}{\partial x} = W^\top \cdot \frac{\partial L}{\partial z}$$

```python
# Forward
z = X @ W + b            # (batch, out_dim)
a = np.maximum(0, z)     # ReLU

# Backward
dL_da = upstream_grad    # (batch, out_dim)
dL_dz = dL_da * (z > 0) # ReLU derivative
dL_dW = X.T @ dL_dz     # (in_dim, out_dim)
dL_db = dL_dz.sum(0)    # (out_dim,)
dL_dX = dL_dz @ W.T     # (batch, in_dim)
```

**Gradient check:** Verify your backprop with finite differences before trusting it:

```python
from scipy.optimize import check_grad

def loss(params):
    # unpack params, compute forward pass, return scalar loss
    ...

def grad(params):
    # your backprop
    ...

error = check_grad(loss, grad, params_init)
assert error < 1e-5, f"Gradient check failed: error = {error:.2e}"
```

---

### 2.3 Hessians and Second-Order Methods

The **Hessian** $H \in \mathbb{R}^{n \times n}$ is the matrix of second partial derivatives:
$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

**Newton's method** uses the Hessian to take curvature-aware steps:
$$x \leftarrow x - H^{-1} \nabla f(x)$$

For convex problems, Newton's method converges in $O(\log \log \frac{1}{\epsilon})$ iterations — far faster than gradient descent's $O(\frac{1}{\epsilon})$. The cost: computing and inverting $H$ is $O(n^3)$, prohibitive for neural networks with millions of parameters.

**Why gradient descent dominates deep learning:** Computing $H^{-1}$ for a network with $10^7$ parameters would require $10^{21}$ FLOPs. Adam approximates the diagonal of $H^{-1}$ cheaply — it's a diagonal quasi-Newton method.

---

## 3. Probability & Information Theory

### 3.1 Maximum Likelihood Estimation

Given data $\mathcal{D} = \{x_1, \ldots, x_n\}$ assumed i.i.d. from $p(x; \theta)$:
$$\hat{\theta}_\text{MLE} = \underset{\theta}{\arg\max} \prod_{i=1}^n p(x_i; \theta) = \underset{\theta}{\arg\max} \sum_{i=1}^n \log p(x_i; \theta)$$

**For linear regression** with Gaussian noise $y = x^\top \beta + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$:
$$\log p(y | x, \beta) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\|y - X\beta\|_2^2$$

Maximizing this over $\beta$ is exactly minimizing $\|y - X\beta\|_2^2$ — MSE is MLE under Gaussian noise.

**For logistic regression** with Bernoulli likelihood:
$$\log p(y | x, w) = \sum_i y_i \log \sigma(x_i^\top w) + (1-y_i)\log(1-\sigma(x_i^\top w))$$

This is the negative binary cross-entropy. So binary cross-entropy minimization = MLE under Bernoulli.

---

### 3.2 KL Divergence & Cross-Entropy

**KL Divergence** (not a metric — asymmetric):
$$D_\text{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} \geq 0$$

Zero iff $P = Q$. Measures "how much information is lost when using $Q$ to approximate $P$."

**Cross-entropy:**
$$H(P, Q) = -\sum_x P(x) \log Q(x) = H(P) + D_\text{KL}(P \| Q)$$

When $P$ is the true label distribution (a one-hot vector), $H(P) = 0$, so minimizing cross-entropy = minimizing KL = making $Q$ approximate $P$ as closely as possible.

```python
def cross_entropy(y_true_onehot, y_pred_probs, eps=1e-12):
    """
    y_true_onehot: (batch, n_classes)
    y_pred_probs:  (batch, n_classes), output of softmax
    """
    y_pred_clipped = np.clip(y_pred_probs, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred_clipped), axis=1))
```

---

### 3.3 Information Gain & Entropy

**Shannon entropy** of discrete variable $X$:
$$H(X) = -\sum_x p(x) \log_2 p(x)$$

Measures "expected surprise" — high entropy = unpredictable. A fair coin has $H = 1$ bit. A biased coin ($p=0.9$) has $H \approx 0.47$ bits.

**Information gain** (used in decision trees):
$$\text{IG}(X, A) = H(X) - \sum_{v \in \text{values}(A)} \frac{|X_v|}{|X|} H(X_v)$$

Measures reduction in entropy after splitting on attribute $A$. Greedy tree building picks the split with maximum IG.

**Log-odds interpretation:** For binary classification, logistic regression models $\log\frac{p}{1-p} = x^\top w$ — the log-odds. Each feature contributes additively to the log-odds, and sigmoid maps it back to a probability.

```python
def entropy(y):
    """Shannon entropy of label array y."""
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-12))

def information_gain(y, y_left, y_right):
    n = len(y)
    return entropy(y) - (len(y_left)/n)*entropy(y_left) - (len(y_right)/n)*entropy(y_right)
```

---

## 4. Quick Reference: Derivatives

| Function | $f(x)$ | Derivative $f'(x)$ | Notes |
|----------|--------|-------------------|-------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | Max at 0.25, approaches 0 for large \|x\| |
| Tanh | $\tanh(x)$ | $1 - \tanh^2(x)$ | Range $(-1,1)$, saturates like sigmoid |
| ReLU | $\max(0, x)$ | $\mathbf{1}[x > 0]$ | Non-differentiable at 0; use subgradient |
| Leaky ReLU | $\max(\alpha x, x)$ | $\alpha$ if $x<0$, else $1$ | Fixes dying ReLU; $\alpha \approx 0.01$ |
| ELU | $x$ if $x>0$, $\alpha(e^x-1)$ if $x\leq 0$ | $1$ if $x>0$, $f(x)+\alpha$ if $x\leq 0$ | Smooth, negative saturation |
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | $\sigma_i(\delta_{ij} - \sigma_j)$ | Jacobian is full matrix; simplifies with cross-entropy |
| MSE | $\frac{1}{n}\|y - \hat{y}\|_2^2$ | $\frac{-2}{n}(y - \hat{y})$ | Sensitive to outliers |
| Cross-Entropy | $-\sum y \log \hat{y}$ | $-y/\hat{y}$ | Paired with softmax: gradient simplifies to $\hat{y} - y$ |
| L2 Regularizer | $\frac{\lambda}{2}\|w\|_2^2$ | $\lambda w$ | Shrinks weights toward 0 |
| L1 Regularizer | $\lambda\|w\|_1$ | $\lambda \cdot \text{sign}(w)$ | Not differentiable at 0; use proximal operator |

**Softmax + Cross-Entropy (combined backward):** The Jacobian of softmax is complex, but the gradient of cross-entropy loss through softmax simplifies beautifully:
$$\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$$

This is the "gift" of using cross-entropy with softmax — implement the backward pass in one line.

---

## 5. Numerical Stability Notes

**Log-sum-exp trick** (prevents softmax overflow):
$$\log \sum_j e^{x_j} = c + \log \sum_j e^{x_j - c}, \quad c = \max_j x_j$$

```python
def softmax(x):
    x_shifted = x - x.max(axis=-1, keepdims=True)  # subtract max
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)
```

**Sigmoid clipping:**
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
```

**Log of probabilities:** Never compute `log(p)` directly — use `log(clip(p, 1e-12, 1))` or compute in log-space throughout.

**OLS near-singular matrices:** Use `np.linalg.lstsq` over `np.linalg.inv(X.T @ X)`. `lstsq` uses SVD internally and handles rank-deficient cases gracefully.

**Gradient explosion in RNNs:** Clip gradients by norm before applying:
```python
norm = np.linalg.norm(grad)
if norm > max_norm:
    grad = grad * (max_norm / norm)
```

---

## 6. Further Reading

| Resource | What to take from it |
|----------|---------------------|
| Bishop, *Pattern Recognition and Machine Learning* (2006) | The definitive probabilistic treatment of ML algorithms |
| Goodfellow et al., *Deep Learning* (2016) | Chapters 2–4 for math foundations, Chapter 6 for backprop |
| Strang, *Linear Algebra and Its Applications* | The clearest explanation of SVD and its geometry |
| Ruder (2016), *An overview of gradient descent algorithms* | When to use SGD vs Adam in practice |
| Bottou et al. (2018), *Optimization Methods for Large-Scale ML* | The rigorous theory behind stochastic optimization |

See [linear_models.md](./linear_models.md) for how these foundations apply directly to OLS and logistic regression derivations.
