# Design Principles

> *Why pure NumPy? Because the best way to understand a machine is to build it with your hands.*

---

## Table of Contents

- [The Core Thesis](#1-the-core-thesis)
- [Why Pure NumPy Is the Ultimate Test](#2-why-pure-numpy-is-the-ultimate-test)
- [Design Tradeoffs](#3-design-tradeoffs)
- [Code Architecture Decisions](#4-code-architecture-decisions)
- [Readability vs Performance](#5-readability-vs-performance)
- [Educational Priorities](#6-educational-priorities)
- [What This Repo Is Not](#7-what-this-repo-is-not)

---

## 1. The Core Thesis

There is a class of ML engineer who can tune hyperparameters, read leaderboards, and ship models to production. This is valuable work. But there is a deeper class who can look at a diverging loss curve and immediately know whether the issue is the learning rate, the initialization, the gradient computation, or a numerical instability — because they have implemented all of these from scratch.

**This repo exists to produce the second kind of engineer.**

When you implement backpropagation yourself — not by calling `loss.backward()` but by computing $\delta^{(l)} = (W^{(l+1)\top}\delta^{(l+1)}) \odot \sigma'(z^{(l)})$ — you confront every design decision explicitly:

- What exactly is a "layer"? What state does it need to store?
- When do I divide by batch size and when don't I?
- Why does the order of operations in Adam matter for bias correction?

These questions have answers that PyTorch handles invisibly. Building from scratch forces you to find those answers.

---

## 2. Why Pure NumPy Is the Ultimate Test

### 2.1 NumPy Exposes Everything

PyTorch's autograd gives you gradients for free. NumPy doesn't. When you implement a backward pass in NumPy, you must:

1. Derive the gradient analytically.
2. Implement it correctly with matrix operations.
3. Verify it against finite differences.

This process is slow and uncomfortable. It is also the most effective way to internalize the chain rule, Jacobians, and the computational structure of neural networks.

### 2.2 The Discipline of Vectorization

Python loops over samples are 100–1000x slower than NumPy's C backend. Forcing yourself to implement batch operations without PyTorch's automagic broadcasting teaches you to think in tensors:

- How does a dense layer's forward pass work for a whole batch? It's `X @ W + b`.
- What shape is the gradient of the loss with respect to `W`? It's `X.T @ dL_dz`.
- How does a convolution operation vectorize? It's a specific `einsum` or `as_strided` call.

Engineers who have never implemented batch operations manually often write subtly wrong vectorized code. You can't fake your way through NumPy — it either broadcasts correctly or it doesn't.

### 2.3 No Black Boxes, No Excuses

Every line of this codebase can be traced back to:
- A mathematical operation (specified in the docstring)
- A gradient derivation (linked in the docs)
- A test (verifying correctness against sklearn or finite differences)

When a model fails, there is nowhere to hide. The bug is in the code you wrote. This is uncomfortable and educational.

---

## 3. Design Tradeoffs

| Decision | We chose | We didn't choose | Why |
|----------|----------|-----------------|-----|
| Computation backend | NumPy | PyTorch / JAX | Transparency; no magic |
| Autodiff | Manual backprop | Autograd / tape | Forces derivation |
| Speed | ~0.6x sklearn | sklearn-parity | Clarity over performance |
| API style | sklearn-like | PyTorch-like | Familiar interface; reduces friction |
| Testing | pytest + finite-diff | just pytest | Gradient correctness is non-negotiable |
| Typing | Full type hints | Untyped | Mypy catches shape bugs before runtime |
| Notebooks | 40+ Jupyter notebooks | Just code | Learning requires narrative |

### Speed Tradeoff

Our implementations run at roughly 0.5x–0.75x the speed of scikit-learn. This is a deliberate tradeoff:

- scikit-learn uses Cython, LAPACK, and BLAS — compiled C/Fortran with hardware-level optimization.
- We use Python and NumPy — readable, auditable, modifiable.

The 2x slowdown is the cost of transparency. For production use, you'd use scikit-learn. For understanding what scikit-learn is doing, you use this.

> **Rule:** We optimize for learning, then for correctness, then for speed. In that order.

---

## 4. Code Architecture Decisions

### 4.1 sklearn-Compatible API

All estimators follow the sklearn convention:
- `fit(X, y)` — trains and returns `self`
- `predict(X)` — inference
- `fit_transform(X)` — for transformers

Why: sklearn's API has become the de facto standard in Python ML. By matching it, our implementations are drop-in compatible with sklearn pipelines, and users can compare outputs directly.

### 4.2 No Inheritance-Heavy Hierarchies

We don't have deep class hierarchies like `BaseEstimator → LinearModel → LinearRegression`. This is intentional. Deep inheritance obscures what's actually happening in each class. Every model is relatively self-contained — you can read `LinearRegression.fit()` and understand exactly what it does without chasing parent class methods.

### 4.3 Explicit Over Implicit

```python
# Bad: implicit, magical
model.compile(optimizer="adam")
model.fit(X, y)

# Good: explicit, traceable
model.fit(X, y, optimizer="adam", lr=1e-3, epochs=100)
```

We prefer function arguments over hidden state. It makes experiments reproducible and code reviewable.

### 4.4 Math-First Variable Names

```python
# Bad: generic names
def forward(self, inputs, w):
    return inputs @ w

# Good: matches the math
def forward(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
    # z = XW + b
    return X @ W + self.b
```

Variable names match the mathematical notation in the docstrings and linked docs. A reader switching between the code and the derivation should never be confused about what a variable represents.

### 4.5 Gradient Checks Are First-Class Citizens

Every backward pass is accompanied by a gradient check in tests. This is not optional:

```python
# tests/test_linear_regression.py
def test_gradient_correctness():
    from scipy.optimize import check_grad
    model = LinearRegression()
    X, y = make_regression(n_samples=20, n_features=5)
    model.fit(X, y, epochs=0)  # just initialize

    def loss(beta): return model._mse(X, y, beta)
    def grad(beta): return model._grad(X, y, beta)

    error = check_grad(loss, grad, model.beta_)
    assert error < 1e-5, f"Gradient error too large: {error:.2e}"
```

---

## 5. Readability vs Performance

Three levels of the same operation — pick the right one for the context:

```python
# Level 1: Educational (use in notebook explanations)
# Shows exactly what's happening
loss = 0
for i in range(n):
    loss += (y[i] - X[i] @ beta) ** 2
loss /= n

# Level 2: Production NumPy (use in library code)
# Vectorized, readable
residuals = y - X @ beta
loss = np.mean(residuals ** 2)

# Level 3: Optimized NumPy (use only in hot paths)
# einsum, as_strided, etc. — fast but opaque
loss = np.einsum('i,i->', residuals, residuals) / n
```

**Policy:**
- Notebooks use Level 1 or 2 — clarity for learners.
- Library code uses Level 2 — vectorized and readable.
- Level 3 only when profiling shows it's necessary, with a comment explaining the operation.

---

## 6. Educational Priorities

### Principle: Every Concept Appears in Three Places

1. **Math doc** (`docs/math_foundations.md`): derivation, intuition, notation
2. **Library code** (`src/`): implementation, typed, tested
3. **Notebook** (`notebooks/`): interactive exploration, visualization, "what happens if I change X?"

If a concept doesn't appear in all three places, it's not fully documented.

### Principle: Fail Loudly and Informatively

```python
def fit(self, X, y, lr=0.01, epochs=1000):
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be np.ndarray, got {type(X)}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
```

Informative error messages are not optional. When a learner makes a shape mistake, the error should tell them exactly what went wrong.

### Principle: Document the Why, Not Just the What

```python
# Bad comment:
# clip to avoid overflow
z_clipped = np.clip(z, -500, 500)

# Good comment:
# exp(x) overflows for x > ~710; clip prevents inf in sigmoid
# 500 is conservative but safe across all float64 hardware
z_clipped = np.clip(z, -500, 500)
```

---

## 7. What This Repo Is Not

**Not a production ML library.** Use scikit-learn, PyTorch, or JAX for production.

**Not an auto-ML framework.** There is no hyperparameter tuning magic — you write the loops.

**Not a research framework.** We don't support distributed training, mixed precision, or custom CUDA kernels.

**Not a shortcut.** If you're looking to skip the math and get a working model quickly, this is the wrong place. If you're looking to understand ML at the level where you could implement it from memory on a whiteboard — this is exactly the right place.

---

*The best debugger is a deep understanding of what the code is supposed to do. Build from scratch. Understand everything.*
