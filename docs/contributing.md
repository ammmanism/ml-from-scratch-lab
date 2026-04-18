# Contributing to ML From Scratch Lab

> *Elite open-source standards. We hold contributions to the same bar as the existing code: mathematically correct, tested, typed, and readable.*

---

## Table of Contents

- [Dev Environment Setup](#1-dev-environment-setup)
- [Code Quality Standards](#2-code-quality-standards)
- [Writing Tests](#3-writing-tests)
- [Gradient Checks](#4-gradient-checks)
- [Adding a New Model](#5-adding-a-new-model)
- [Documentation Standards](#6-documentation-standards)
- [Pull Request Process](#7-pull-request-process)
- [Code of Conduct](#8-code-of-conduct)

---

## 1. Dev Environment Setup

```bash
git clone https://github.com/ammmanism/ml-from-scratch-lab.git
cd ml-from-scratch-lab

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -e ".[dev]"
pre-commit install
```

**Verify setup:**
```bash
pytest tests/ -q           # all tests pass
black --check src/ tests/  # no formatting issues
flake8 src/ tests/         # no linting errors
mypy src/                  # no type errors
```

**Pre-commit hooks run automatically on `git commit`:**
- `black` — formatting
- `flake8` — linting
- `mypy` — type checking
- `pytest` (fast tests only, marked `@pytest.mark.fast`)

---

## 2. Code Quality Standards

### Formatting: Black

Line length 88. No config needed — just run:
```bash
black src/ tests/
```

### Linting: Flake8

```bash
flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
```

### Type Safety: Mypy

All public functions must have type annotations. Run:
```bash
mypy src/ --ignore-missing-imports --strict
```

**Required for all public APIs:**
```python
def fit(
    self,
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    epochs: int = 1000,
) -> "LinearRegression":  # return self for chaining
    ...
```

### Docstrings: Google Format

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    """Generate predictions for input samples.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
            Must be the same feature space as training data.

    Returns:
        Predicted values of shape (n_samples,).

    Raises:
        RuntimeError: If model has not been fitted yet.

    Example:
        >>> model = LinearRegression()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
    """
```

---

## 3. Writing Tests

### Structure

```
tests/
├── test_linear_regression.py
├── test_logistic_regression.py
├── test_decision_tree.py
├── test_kmeans.py
├── test_pca.py
├── test_mlp.py
├── test_gradients.py       ← gradient checks live here
└── conftest.py             ← shared fixtures
```

### Test Categories

Mark tests appropriately:

```python
import pytest

@pytest.mark.fast           # runs in pre-commit (<1s)
def test_shapes():
    ...

@pytest.mark.slow           # runs in CI only
def test_mnist_accuracy():
    ...

@pytest.mark.gradient       # gradient check tests
def test_backprop_dense():
    ...
```

### Required Tests for Every Model

```python
def test_fit_predict_shapes():
    """Output shapes are correct."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)

def test_perfect_fit_on_trivial_data():
    """Model achieves near-perfect accuracy on linearly separable data."""
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    model.fit(X, y, epochs=5000)
    assert (model.predict(X) == y).all()

def test_matches_sklearn():
    """Our implementation matches sklearn within tolerance."""
    from sklearn.linear_model import LinearRegression as SKLearnLR
    sk_model = SKLearnLR()
    sk_model.fit(X_train, y_train)
    our_model = LinearRegression(solver="closed_form")
    our_model.fit(X_train, y_train)
    np.testing.assert_allclose(
        our_model.predict(X_test),
        sk_model.predict(X_test),
        rtol=1e-4,
    )

def test_gradient_correctness():
    """Analytical gradient matches finite differences."""
    # See Section 4 below
    ...
```

### Fixtures (conftest.py)

```python
import pytest
import numpy as np
from ml_from_scratch.utils import make_regression, make_classification

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    return X, y

@pytest.fixture
def binary_classification_data():
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    return X, y
```

---

## 4. Gradient Checks

**Every backward pass must be tested against finite differences.** This is non-negotiable.

### Using `scipy.optimize.check_grad`

```python
from scipy.optimize import check_grad
import numpy as np

def test_linear_regression_gradient(regression_data):
    from ml_from_scratch.linear import LinearRegression

    X, y = regression_data
    model = LinearRegression()

    def loss_fn(beta):
        return np.mean((y - X @ beta[:-1] - beta[-1]) ** 2)

    def grad_fn(beta):
        return model._compute_gradient(X, y, beta)

    beta_init = np.random.randn(X.shape[1] + 1)
    error = check_grad(loss_fn, grad_fn, beta_init)
    assert error < 1e-5, f"Gradient error: {error:.2e}"
```

### Manual Finite Difference Check

```python
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

def test_mlp_backward():
    """MLP backward pass matches numerical gradient."""
    model = MLP(layers=[4, 8, 3])
    X = np.random.randn(5, 4)
    y = np.eye(3)[[0,1,2,0,1]]

    analytical_grads = model._backward(X, y)
    numerical_grads = numerical_gradient(lambda p: model._loss(X, y, p), model._flatten_params())

    for ag, ng in zip(analytical_grads, numerical_grads):
        rel_error = np.abs(ag - ng) / (np.abs(ag) + np.abs(ng) + 1e-8)
        assert rel_error.max() < 1e-4
```

---

## 5. Adding a New Model

### Checklist

- [ ] Implement in the correct submodule (`linear/`, `trees/`, `neural/`, `unsupervised/`, `advanced/`)
- [ ] Inherit from nothing (keep it self-contained) or from a minimal `BaseEstimator`
- [ ] Implement `fit(X, y) -> self` and `predict(X) -> np.ndarray`
- [ ] Full type annotations on all public methods
- [ ] Google-style docstring on every public method
- [ ] Unit tests: shape, trivial data, sklearn comparison
- [ ] Gradient check test (if model has gradients)
- [ ] Export from `ml_from_scratch/__init__.py`
- [ ] Add to `docs/api_reference.md`
- [ ] Add a notebook in `notebooks/` demonstrating usage with visualizations
- [ ] Add to learning path in `README.md` if appropriate
- [ ] Run full test suite: `pytest tests/ --cov=ml_from_scratch`

### File Template

```python
# src/ml_from_scratch/linear/new_model.py
"""Brief one-line description.

Mathematical background: See docs/linear_models.md#section-name.
"""

from __future__ import annotations

import numpy as np


class NewModel:
    """One-sentence class description.

    Implements [algorithm name] via [method].
    See: [link to derivation in docs].

    Args:
        param1: Description. Default: value.
        param2: Description. Default: value.

    Attributes:
        param1_: Description (set after fit).

    Example:
        >>> model = NewModel(param1=value)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
    """

    def __init__(
        self,
        param1: float = 1.0,
        fit_intercept: bool = True,
    ) -> None:
        self.param1 = param1
        self.fit_intercept = fit_intercept
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 1000,
    ) -> "NewModel":
        """Fit the model to training data.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Targets, shape (n_samples,).
            lr: Learning rate for gradient descent.
            epochs: Number of gradient steps.

        Returns:
            self — for method chaining.
        """
        self._validate_input(X, y)
        # ... implementation ...
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Predictions, shape (n_samples,).

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        # ... implementation ...

    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be np.ndarray, got {type(X)}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
```

---

## 6. Documentation Standards

### Mathematical Notation

Use LaTeX in docstrings and markdown. Always define variables:

```
The gradient of MSE with respect to β is:

    ∇_β L = (2/n) Xᵀ(Xβ - y)

where X ∈ ℝⁿˣᵖ is the design matrix and y ∈ ℝⁿ is the target vector.
```

### Cross-References

Always link between docs:
- Code docstrings → `docs/` for derivations
- `docs/api_reference.md` → implementation files
- `docs/math_foundations.md` ↔ `docs/linear_models.md`

### Notebooks

Every new notebook must:
1. Start with a "Learning objectives" cell
2. Include visualizations (decision boundaries, loss curves, etc.)
3. End with a "Key Takeaways" summary cell
4. Be verified to execute clean (`pytest --nbval notebooks/`)

---

## 7. Pull Request Process

1. **Open an issue first** for non-trivial changes. Get alignment before investing significant time.

2. **Branch naming:**
   - `feat/naive-bayes` — new model
   - `fix/softmax-overflow` — bug fix
   - `docs/api-reference-update` — docs only
   - `perf/vectorize-convolution` — performance

3. **PR must pass:**
   - All existing tests
   - New tests for your changes
   - `black`, `flake8`, `mypy`
   - Coverage does not decrease

4. **PR description must include:**
   - What changed and why
   - Mathematical reference (paper, section of textbook) if applicable
   - Benchmark impact if performance-related
   - "Breaking change" section if API changed

5. **Review turnaround:** maintainers aim to review within 72 hours.

---

## 8. Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Be excellent to each other. Critique code, not people. Mathematical disagreements should be resolved with derivations and tests, not authority.
