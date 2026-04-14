# 📐 Linear Models: Deep Dive into Optimization

Linear models are the "atomic units" of machine learning. In this repository, we move beyond simple `dot` products to explore the optimization regimes that make these models converge in high-dimensional space.

---

## 🏗️ Table of Contents
1. [Ordinary Least Squares: The Projection Intuition](#1-ordinary-least-squares-the-projection-intuition)
2. [Logistic Regression: Log-Odds and Decision Boundaries](#2-logistic-regression-log-odds-and-decision-boundaries)
3. [Lasso and the Soft-Thresholding Operator](#3-lasso-and-the-soft-thresholding-operator)
4. [Proximal Gradient Descent](#4-proximal-gradient-descent)

---

## 1. Ordinary Least Squares: The Projection Intuition

### The Geometric View
In OLS, we have $X\theta = y$. If $y$ is not in the column space of $X$, there is no solution. We seek the $\hat{y}$ that is the **orthogonal projection** of $y$ onto the space spanned by $X$.
- This projection is mathematically equivalent to solving the **Normal Equation**:
  $$\theta = (X^T X)^{-1} X^T y$$
- **Stability Note**: If $X$ has redundant features (collinearity), $X^T X$ becomes singular. We resolve this by adding a "Tikhonov" regularizer $\lambda I$.

---

## 2. Logistic Regression: Log-Odds and Decision Boundaries

### From Linear to Probabilistic
We model the log-odds (logit) as a linear combination of inputs:
$$\log\left(\frac{p}{1-p}\right) = X\theta$$
Solving for $p$, we arrive at the **Sigmoid Function**:
$$p = \frac{1}{1 + e^{-X\theta}}$$

### Maximum Likelihood Estimation (MLE)
We find $\theta$ by minimizing the Negative Log-Likelihood (Binary Cross-Entropy):
$$J(\theta) = -\frac{1}{m} \sum [y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]$$
**Implementation Challenge**: `log(0)` is undefined. Our implementation handles this by clipping $\hat{y}$ to $[\epsilon, 1-\epsilon]$.

---

## 3. Lasso and the Soft-Thresholding Operator

Unlike Ridge (L2), Lasso (L1) creates **sparsity**. Because the L1-norm $||\theta||_1$ is not differentiable at $\theta=0$, we cannot use standard Gradient Descent.

### The Soft-Thresholding Function
The analytical solution for a coordinate-wise update in Lasso:
$$S_\lambda(z) = \text{sign}(z)(|z| - \lambda)_+$$
Where $(x)_+$ is the hinge function $\max(0, x)$.
- **Intuition**: We "pull" the weights toward zero. If the signal is weaker than the penalty $\lambda$, the weight is snapped exactly to zero. This is the "Automatic Feature Selection" property of Lasso.

---

## 4. Proximal Gradient Descent

For models with both differentiable and non-differentiable parts (like ElasticNet), we use **Proximal Gradient Descent**.
1. **The Gradient Step**: Take a step down the smooth part (the standard loss).
2. **The Proximal Step**: Apply the Soft-Thresholding operator to the result.

This ensures we reach the true sparse minimum without numerical oscillations.

---

## 🧠 Historical Note
> **Ronald Fisher** first derived Linear Discriminant Analysis in 1936. While it assumes Gaussian features, it remains one of the most robust "Lite" models for high-dimensional text classification even today.

---

## 💡 Engineering Pitfall: Multicollinearity
When two features are perfectly correlated, the loss landscape of OLS is not a bowl but a **flat-bottomed valley**. There are infinitely many "best" solutions. **Ridge Regularization** is the specific tool used to turn this valley back into a bowl with a single, stable minimum.
