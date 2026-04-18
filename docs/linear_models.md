# Linear Models: Derivations & Deep Dives

> *If you can derive OLS from scratch and implement it without looking anything up, you understand linear models. This document gets you there.*

---

## Table of Contents

- [Ordinary Least Squares (OLS)](#1-ordinary-least-squares-ols)
  - [Problem Setup](#11-problem-setup)
  - [Closed-Form Derivation](#12-closed-form-derivation)
  - [Geometric Interpretation](#13-geometric-interpretation)
  - [Gradient Descent Derivation](#14-gradient-descent-derivation)
- [Logistic Regression](#2-logistic-regression)
  - [Why Not Linear Regression for Classification?](#21-why-not-linear-regression-for-classification)
  - [The Logit and Sigmoid](#22-the-logit-and-sigmoid)
  - [Cross-Entropy Loss Derivation](#23-cross-entropy-loss-derivation)
  - [Gradient of the Log-Likelihood](#24-gradient-of-the-log-likelihood)
  - [Multiclass: Softmax Regression](#25-multiclass-softmax-regression)
- [Regularization](#3-regularization)
  - [Ridge (L2)](#31-ridge-l2)
  - [Lasso (L1) and the Proximal Operator](#32-lasso-l1-and-the-proximal-operator)
  - [ElasticNet](#33-elasticnet)
  - [Geometric Intuition](#34-geometric-intuition)
- [Numerical Implementation Notes](#4-numerical-implementation-notes)
- [Connection to Probabilistic Models](#5-connection-to-probabilistic-models)

---

## 1. Ordinary Least Squares (OLS)

### 1.1 Problem Setup

Given:
- Design matrix $X \in \mathbb{R}^{n \times p}$ ($n$ samples, $p$ features, **with bias column**)
- Target vector $y \in \mathbb{R}^n$

Find $\hat{\beta} \in \mathbb{R}^p$ that minimizes the **residual sum of squares**:

$$\mathcal{L}(\beta) = \|y - X\beta\|_2^2 = \sum_{i=1}^n (y_i - x_i^\top \beta)^2$$

---

### 1.2 Closed-Form Derivation

Expand the loss:
$$\mathcal{L}(\beta) = (y - X\beta)^\top(y - X\beta) = y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X \beta$$

Take the derivative with respect to $\beta$ and set to zero:
$$\frac{\partial \mathcal{L}}{\partial \beta} = -2X^\top y + 2X^\top X \beta = 0$$

$$\boxed{\hat{\beta} = (X^\top X)^{-1} X^\top y}$$

This is the **Normal Equation**. It exists and is unique when $X^\top X$ is invertible (i.e., $X$ has full column rank, which requires $n \geq p$ and no collinear features).

```python
def fit_closed_form(X, y):
    # Add bias column
    X_b = np.hstack([X, np.ones((len(X), 1))])
    # Use lstsq for numerical stability (SVD-based, handles rank-deficient cases)
    beta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
    return beta

# Or explicitly (only when X is well-conditioned):
# beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
```

> **Warning:** Never blindly invert $X^\top X$. If features are nearly collinear, the matrix is nearly singular and the solution explodes. Use `np.linalg.lstsq` (SVD-based) or regularize.

---

### 1.3 Geometric Interpretation

$\hat{y} = X\hat{\beta}$ is the **orthogonal projection** of $y$ onto the column space of $X$. The residuals $e = y - \hat{y}$ are perpendicular to every column of $X$:

$$X^\top e = X^\top(y - X\hat{\beta}) = 0$$

This is exactly the Normal Equation. OLS finds the closest point in the linear subspace spanned by the features — "closest" in Euclidean (L2) distance.

**Hat matrix:** $\hat{y} = Hy$ where $H = X(X^\top X)^{-1}X^\top$ is the projection matrix. Note $H^2 = H$ (idempotent) and $H = H^\top$ (symmetric) — properties of all projection matrices.

---

### 1.4 Gradient Descent Derivation

When $n$ or $p$ is large, the normal equation is $O(np^2 + p^3)$ — too slow. We use gradient descent.

Gradient of MSE loss $\mathcal{L} = \frac{1}{n}\|y - X\beta\|_2^2$:

$$\nabla_\beta \mathcal{L} = \frac{-2}{n} X^\top (y - X\beta) = \frac{2}{n} X^\top (X\beta - y)$$

Update rule:
$$\beta \leftarrow \beta - \alpha \nabla_\beta \mathcal{L}$$

```python
def fit_gradient_descent(X, y, lr=0.01, epochs=1000):
    X_b = np.hstack([X, np.ones((len(X), 1))])
    n, p = X_b.shape
    beta = np.zeros(p)  # or He init

    for _ in range(epochs):
        residuals = X_b @ beta - y           # (n,)
        grad = (2/n) * X_b.T @ residuals    # (p,)
        beta -= lr * grad

    return beta
```

**Convergence condition:** Gradient descent on MSE converges when $\alpha < \frac{2}{\lambda_\max(X^\top X / n)}$. The condition number $\kappa(X^\top X)$ determines how many iterations you need — high condition number (nearly collinear features) → many iterations → normalize your features.

---

## 2. Logistic Regression

### 2.1 Why Not Linear Regression for Classification?

Linear regression on binary labels ($y \in \{0,1\}$) produces unbounded outputs — probabilities outside $[0,1]$, undefined for cross-entropy. It also assumes homoscedastic Gaussian noise, which is wrong for classification.

More fundamentally: the MSE loss on binary labels is not the right objective. We want to maximize the **likelihood** of the observed binary outcomes.

---

### 2.2 The Logit and Sigmoid

We want $P(y=1|x)$ to be a function of $x^\top w$ that lies in $(0,1)$. The key insight: model the **log-odds**:

$$\log \frac{P(y=1|x)}{P(y=0|x)} = x^\top w$$

Solving for $P(y=1|x)$:

$$P(y=1|x) = \sigma(x^\top w) = \frac{1}{1 + e^{-x^\top w}}$$

The sigmoid $\sigma$ maps $\mathbb{R} \to (0,1)$, and its derivative has the elegant form:
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
```

---

### 2.3 Cross-Entropy Loss Derivation

Assume Bernoulli likelihood: $p(y_i | x_i, w) = \hat{y}_i^{y_i}(1-\hat{y}_i)^{1-y_i}$ where $\hat{y}_i = \sigma(x_i^\top w)$.

Log-likelihood:
$$\ell(w) = \sum_{i=1}^n \left[ y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i) \right]$$

Negating (convert max to min):
$$\mathcal{L}(w) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i) \right]$$

This is **binary cross-entropy**, and it directly follows from MLE — no arbitrary choice of loss function.

```python
def binary_cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

---

### 2.4 Gradient of the Log-Likelihood

Differentiating $\mathcal{L}$ with respect to $w$:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i) x_i = \frac{1}{n} X^\top (\hat{y} - y)$$

This has the same form as OLS! The difference: $\hat{y} = \sigma(Xw)$ instead of $\hat{y} = Xw$.

```python
def fit(X, y, lr=0.1, epochs=500):
    X_b = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X_b.shape[1])

    for _ in range(epochs):
        y_hat = sigmoid(X_b @ w)             # (n,)
        grad = (1/len(y)) * X_b.T @ (y_hat - y)  # (p+1,)
        w -= lr * grad

    return w
```

> **Note:** Logistic regression has no closed-form solution (unlike OLS) because $\hat{y}$ is nonlinear in $w$. But the loss is **convex**, so gradient descent converges to the global minimum.

---

### 2.5 Multiclass: Softmax Regression

For $K$ classes, model:
$$P(y=k|x) = \frac{e^{x^\top w_k}}{\sum_{j=1}^K e^{x^\top w_j}} = \text{softmax}(Xw)_k$$

Weight matrix $W \in \mathbb{R}^{p \times K}$.

Loss (categorical cross-entropy):
$$\mathcal{L}(W) = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K y_{ik} \log \hat{y}_{ik}$$

Gradient (the "gift" of softmax + cross-entropy):
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{n} X^\top (\hat{Y} - Y)$$

where $\hat{Y} \in \mathbb{R}^{n \times K}$ is the softmax output and $Y$ is the one-hot target matrix.

```python
def softmax(z):
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def categorical_cross_entropy(Y_onehot, Y_pred, eps=1e-12):
    return -np.mean(np.sum(Y_onehot * np.log(np.clip(Y_pred, eps, 1)), axis=1))

def fit_multiclass(X, Y_onehot, lr=0.1, epochs=500):
    n, p = X.shape
    K = Y_onehot.shape[1]
    W = np.zeros((p, K))

    for _ in range(epochs):
        Y_hat = softmax(X @ W)              # (n, K)
        grad = (1/n) * X.T @ (Y_hat - Y_onehot)  # (p, K)
        W -= lr * grad

    return W
```

---

## 3. Regularization

### 3.1 Ridge (L2)

Add L2 penalty to OLS:
$$\mathcal{L}(\beta) = \|y - X\beta\|_2^2 + \lambda\|\beta\|_2^2$$

Gradient:
$$\nabla_\beta \mathcal{L} = -2X^\top y + 2(X^\top X + \lambda I)\beta$$

**Closed-form Ridge:**
$$\hat{\beta}_\text{ridge} = (X^\top X + \lambda I)^{-1} X^\top y$$

The $\lambda I$ term ensures invertibility even when $X^\top X$ is singular — Ridge regression is always uniquely solvable. This is the primary motivation beyond regularization.

**Bayesian interpretation:** Ridge = MLE with a Gaussian prior $p(\beta) = \mathcal{N}(0, \frac{\sigma^2}{\lambda}I)$. The $\lambda$ controls prior tightness.

```python
def fit_ridge(X, y, lam=1.0):
    X_b = np.hstack([X, np.ones((len(X), 1))])
    p = X_b.shape[1]
    # Don't regularize bias: set last diagonal element to 0
    reg = lam * np.eye(p)
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(X_b.T @ X_b + reg, X_b.T @ y)
    return beta
```

---

### 3.2 Lasso (L1) and the Proximal Operator

$$\mathcal{L}(\beta) = \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda\|\beta\|_1$$

The L1 penalty is **not differentiable** at $\beta_j = 0$. No closed-form solution exists. Two approaches:

**Subgradient descent** (simple but slow):
```python
sign_beta = np.sign(beta)
sign_beta[beta == 0] = 0  # subgradient at 0 is any value in [-1, 1]
grad = (1/n) * X.T @ (X @ beta - y) + lam * sign_beta
beta -= lr * grad
```

**Coordinate descent with soft-thresholding (faster):**

For each coordinate $j$, the update has an analytical solution — the **soft-threshold operator** (proximal operator of L1):

$$\hat{\beta}_j \leftarrow \mathcal{S}_{\lambda t}(\beta_j + t r_j) \quad \text{where} \quad \mathcal{S}_\tau(z) = \text{sign}(z)\max(|z| - \tau, 0)$$

```python
def soft_threshold(z, tau):
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0)

def fit_lasso_coordinate_descent(X, y, lam=0.1, epochs=1000):
    n, p = X.shape
    beta = np.zeros(p)
    
    for _ in range(epochs):
        for j in range(p):
            residual = y - X @ beta + X[:, j] * beta[j]  # leave-one-out residual
            rho_j = X[:, j] @ residual
            z_j = np.sum(X[:, j]**2)
            beta[j] = soft_threshold(rho_j / z_j, lam / z_j * n)
    
    return beta
```

**Why Lasso produces sparsity:** The L1 unit ball is diamond-shaped with corners on the axes. The loss contours (ellipses) are most likely to touch a corner — where one or more $\beta_j = 0$ exactly. L2's circular unit ball has no corners, so it shrinks weights uniformly but rarely zeros them.

**Bayesian interpretation:** Lasso = MLE with Laplace prior $p(\beta) \propto e^{-\lambda|\beta|}$.

---

### 3.3 ElasticNet

Combines L1 and L2:
$$\mathcal{L}(\beta) = \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda\left[\alpha\|\beta\|_1 + \frac{1-\alpha}{2}\|\beta\|_2^2\right]$$

- $\alpha = 1$: pure Lasso
- $\alpha = 0$: pure Ridge
- $0 < \alpha < 1$: ElasticNet

**Advantage over Lasso:** When features are correlated, Lasso tends to pick one and drop others arbitrarily. ElasticNet's L2 term groups correlated features together.

---

### 3.4 Geometric Intuition

```
L2 (Ridge)           L1 (Lasso)
    ┌───────┐             ◆
   ╱         ╲          ╱ ╲
  │     ●     │        ╱   ╲
   ╲         ╱        ╱     ╲
    └───────┘         ◆─────◆
                   corners promote sparsity
```

The optimal $\hat{\beta}$ lies where the loss contour first touches the constraint set:
- L2: smooth ball → touches anywhere → small but nonzero weights
- L1: diamond → touches a corner → exact zeros

---

## 4. Numerical Implementation Notes

| Issue | Problem | Fix |
|-------|---------|-----|
| `inv(X.T @ X)` | Singular when features are collinear | Use `lstsq` (SVD-based) |
| Learning rate too large | Loss diverges | Use line search or `lr < 1/lambda_max(X.T @ X)` |
| Not normalizing features | Ill-conditioned gradient, slow convergence | `StandardScaler` before fitting |
| Regularizing the bias | Bias shouldn't be penalized (it's not a weight) | Set `reg[-1,-1] = 0` in Ridge |
| L1 gradient at 0 | Undefined derivative | Use subgradient (0) or coordinate descent |
| Loss plateau | Learning rate too small | Use Adam or line search |

---

## 5. Connection to Probabilistic Models

| Model | Likelihood | Prior | Posterior / Loss |
|-------|-----------|-------|-----------------|
| OLS | $\mathcal{N}(y; X\beta, \sigma^2)$ | None (MLE) | MSE |
| Ridge | $\mathcal{N}(y; X\beta, \sigma^2)$ | $\mathcal{N}(\beta; 0, \tau^2)$ | MSE + L2 |
| Lasso | $\mathcal{N}(y; X\beta, \sigma^2)$ | Laplace$(\beta; 0, b)$ | MSE + L1 |
| Logistic | Bernoulli$(y; \sigma(Xw))$ | None (MLE) | Binary cross-entropy |
| Logistic + L2 | Bernoulli$(y; \sigma(Xw))$ | $\mathcal{N}(w; 0, \tau^2)$ | BCE + L2 |

**Key insight:** Every regularizer is a prior. Choosing a regularizer is equivalent to choosing a Bayesian prior over your model parameters. L2 says "I believe weights are normally distributed around zero." L1 says "I believe most weights are exactly zero (sparse model)."

---

*See [api_reference.md](./api_reference.md) for full class signatures and [math_foundations.md](./math_foundations.md) for the underlying calculus.*
