# API Reference

> Full technical specification for `ml_from_scratch`. Every public class and method with type hints, parameter descriptions, return types, complexity, and usage examples.

---

## Table of Contents

- [Linear Models](#linear-models)
  - [LinearRegression](#linearregression)
  - [LogisticRegression](#logisticregression)
  - [RidgeRegression](#ridgeregression)
  - [LassoRegression](#lassoregression)
- [Tree Models](#tree-models)
  - [DecisionTreeClassifier](#decisiontreeclassifier)
  - [DecisionTreeRegressor](#decisiontreeregressor)
  - [RandomForestClassifier](#randomforestclassifier)
- [Unsupervised](#unsupervised)
  - [KMeans](#kmeans)
  - [PCA](#pca)
- [Neural](#neural)
  - [MLP](#mlp)
  - [Activations](#activations)
  - [Losses](#losses)
- [Utilities](#utilities)
  - [Data Generation](#data-generation)
  - [Metrics](#metrics)
  - [Preprocessing](#preprocessing)

---

## Linear Models

```python
from ml_from_scratch.linear import LinearRegression, LogisticRegression, RidgeRegression, LassoRegression
```

---

### `LinearRegression`

Ordinary least squares linear regression via gradient descent or closed-form normal equation.

**Class signature:**
```python
class LinearRegression:
    def __init__(
        self,
        solver: str = "gradient_descent",  # "gradient_descent" | "closed_form"
        fit_intercept: bool = True,
    ) -> None
```

#### `fit`
```python
def fit(
    self,
    X: np.ndarray,          # shape (n_samples, n_features)
    y: np.ndarray,          # shape (n_samples,)
    lr: float = 0.01,       # learning rate (ignored for closed_form)
    epochs: int = 1000,     # iterations (ignored for closed_form)
    verbose: bool = False,  # print loss every 100 epochs
) -> "LinearRegression"
```

**Complexity:**
- Closed-form: $O(np^2 + p^3)$ — dominated by matrix inversion
- Gradient descent: $O(np \cdot \text{epochs})$ — preferred for large $n, p$

#### `predict`
```python
def predict(self, X: np.ndarray) -> np.ndarray
# Returns: shape (n_samples,), float64
```

#### Attributes (post-fit):
| Attribute | Type | Description |
|-----------|------|-------------|
| `coef_` | `np.ndarray` shape `(n_features,)` | Learned weights |
| `intercept_` | `float` | Bias term |
| `loss_history_` | `list[float]` | MSE per epoch (gradient_descent only) |

**Example:**
```python
from ml_from_scratch.linear import LinearRegression
from ml_from_scratch.utils import make_regression, train_test_split

X, y = make_regression(n_samples=200, n_features=5, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression(solver="gradient_descent")
model.fit(X_train, y_train, lr=0.01, epochs=2000)

preds = model.predict(X_test)
r2 = 1 - ((y_test - preds)**2).sum() / ((y_test - y_test.mean())**2).sum()
print(f"R²: {r2:.4f}")
print(f"Coefs: {model.coef_}")
```

> **See also:** [linear_models.md — OLS derivation](./linear_models.md#1-ordinary-least-squares-ols)

---

### `LogisticRegression`

Binary and multiclass logistic regression via gradient descent. Uses sigmoid (binary) or softmax (multiclass).

```python
class LogisticRegression:
    def __init__(
        self,
        regularization: str | None = None,  # None | "l1" | "l2"
        C: float = 1.0,                     # inverse regularization strength (higher = less reg)
        fit_intercept: bool = True,
        multi_class: str = "auto",          # "auto" | "binary" | "multinomial"
    ) -> None
```

#### `fit`
```python
def fit(
    self,
    X: np.ndarray,       # shape (n_samples, n_features)
    y: np.ndarray,       # shape (n_samples,), integer class labels
    lr: float = 0.1,
    epochs: int = 500,
    batch_size: int | None = None,  # None = full-batch GD
    verbose: bool = False,
) -> "LogisticRegression"
```

**Complexity:** $O(np \cdot \text{epochs})$ per class for mini-batch SGD.

#### `predict`
```python
def predict(self, X: np.ndarray) -> np.ndarray
# Returns: shape (n_samples,), integer class labels
```

#### `predict_proba`
```python
def predict_proba(self, X: np.ndarray) -> np.ndarray
# Returns: shape (n_samples, n_classes), probabilities summing to 1
```

**Example:**
```python
from ml_from_scratch.linear import LogisticRegression
from sklearn.datasets import load_iris
from ml_from_scratch.utils import accuracy_score, train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(regularization="l2", C=10.0, multi_class="multinomial")
model.fit(X_train, y_train, lr=0.1, epochs=1000)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
probs = model.predict_proba(X_test)  # shape (30, 3)
```

---

### `RidgeRegression`

L2-penalized OLS with analytical closed-form solution.

```python
class RidgeRegression:
    def __init__(
        self,
        alpha: float = 1.0,         # regularization strength λ
        fit_intercept: bool = True,
        solver: str = "cholesky",   # "cholesky" | "svd"
    ) -> None
```

**Complexity:** $O(np^2 + p^3)$ — same as OLS but guaranteed invertible.

**Example:**
```python
from ml_from_scratch.linear import RidgeRegression

model = RidgeRegression(alpha=10.0)
model.fit(X_train, y_train)
print(model.coef_)
```

---

### `LassoRegression`

L1-penalized regression via coordinate descent with soft-thresholding.

```python
class LassoRegression:
    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,          # coordinate descent convergence tolerance
        fit_intercept: bool = True,
    ) -> None
```

**Complexity:** $O(np \cdot \text{max\_iter})$ per coordinate sweep.

**Example:**
```python
from ml_from_scratch.linear import LassoRegression

model = LassoRegression(alpha=0.1)
model.fit(X_train, y_train)

n_nonzero = (model.coef_ != 0).sum()
print(f"Sparsity: {n_nonzero}/{len(model.coef_)} features selected")
```

> **See also:** [linear_models.md — Lasso proximal operator](./linear_models.md#32-lasso-l1-and-the-proximal-operator)

---

## Tree Models

```python
from ml_from_scratch.trees import DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier
```

---

### `DecisionTreeClassifier`

CART decision tree for classification. Supports Gini impurity and entropy criterion.

```python
class DecisionTreeClassifier:
    def __init__(
        self,
        criterion: str = "gini",    # "gini" | "entropy"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = None,  # int | "sqrt" | "log2" | None
        random_state: int | None = None,
    ) -> None
```

#### `fit`
```python
def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier"
# Time: O(n p log n) for building; Space: O(n log n) for balanced tree
```

#### `predict`
```python
def predict(self, X: np.ndarray) -> np.ndarray          # integer labels
def predict_proba(self, X: np.ndarray) -> np.ndarray    # leaf class frequencies
```

#### Attributes:
| Attribute | Type | Description |
|-----------|------|-------------|
| `feature_importances_` | `np.ndarray (n_features,)` | Gini-based importance |
| `n_leaves_` | `int` | Number of leaf nodes |
| `depth_` | `int` | Actual depth of fitted tree |

**Example:**
```python
from ml_from_scratch.trees import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
clf.fit(X, y)

preds = clf.predict(X)
print("Train accuracy:", (preds == y).mean())
print("Top feature:", X.columns[clf.feature_importances_.argmax()])
```

---

### `RandomForestClassifier`

Ensemble of decision trees with bootstrap sampling and feature subsampling.

```python
class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: str | int | float = "sqrt",
        min_samples_split: int = 2,
        bootstrap: bool = True,
        oob_score: bool = False,        # compute out-of-bag accuracy
        n_jobs: int = 1,                # -1 = all cores
        random_state: int | None = None,
    ) -> None
```

#### Attributes (post-fit):
| Attribute | Type | Description |
|-----------|------|-------------|
| `estimators_` | `list[DecisionTreeClassifier]` | Fitted trees |
| `feature_importances_` | `np.ndarray` | Mean decrease in impurity across trees |
| `oob_score_` | `float` | OOB accuracy (if `oob_score=True`) |

**Example:**
```python
from ml_from_scratch.trees import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, max_depth=10, oob_score=True, random_state=0)
rf.fit(X_train, y_train)

print("OOB score:", rf.oob_score_)
print("Test acc:", (rf.predict(X_test) == y_test).mean())
```

---

## Unsupervised

```python
from ml_from_scratch.unsupervised import KMeans, PCA
```

---

### `KMeans`

Lloyd's algorithm with k-means++ initialization.

```python
class KMeans:
    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "kmeans++",     # "kmeans++" | "random"
        max_iter: int = 300,
        tol: float = 1e-4,          # centroid shift convergence threshold
        n_init: int = 10,           # number of restarts (best inertia kept)
        random_state: int | None = None,
    ) -> None
```

#### `fit`
```python
def fit(self, X: np.ndarray) -> "KMeans"
# Time: O(n_clusters * n * n_features * max_iter * n_init)
```

#### `predict`
```python
def predict(self, X: np.ndarray) -> np.ndarray
# Returns: shape (n_samples,), integer cluster labels
```

#### `fit_predict`
```python
def fit_predict(self, X: np.ndarray) -> np.ndarray
```

#### Attributes:
| Attribute | Type | Description |
|-----------|------|-------------|
| `cluster_centers_` | `np.ndarray (n_clusters, n_features)` | Final centroids |
| `inertia_` | `float` | Sum of squared distances to nearest centroid |
| `n_iter_` | `int` | Iterations run in best restart |

**Example:**
```python
from ml_from_scratch.unsupervised import KMeans
from sklearn.datasets import load_iris

X, y_true = load_iris(return_X_y=True)
km = KMeans(n_clusters=3, init="kmeans++", random_state=42)
labels = km.fit_predict(X)

print("Inertia:", km.inertia_)
# Evaluate with external label: adjusted rand index
from sklearn.metrics import adjusted_rand_score
print("ARI:", adjusted_rand_score(y_true, labels))
```

---

### `PCA`

SVD-based principal component analysis.

```python
class PCA:
    def __init__(
        self,
        n_components: int | float | str | None = None,
        # int: exact number of components
        # float in (0,1): min explained variance ratio to retain
        # "mle": Minka's MLE
        # None: all components
        whiten: bool = False,   # scale components to unit variance
        random_state: int | None = None,
    ) -> None
```

#### `fit`
```python
def fit(self, X: np.ndarray) -> "PCA"
# Time: O(min(n,p)^2 * max(n,p)) for full SVD
```

#### `transform`
```python
def transform(self, X: np.ndarray) -> np.ndarray
# Returns: shape (n_samples, n_components), projected data
```

#### `fit_transform`
```python
def fit_transform(self, X: np.ndarray) -> np.ndarray
```

#### `inverse_transform`
```python
def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray
# Returns: shape (n_samples, n_features), approximate reconstruction
```

#### Attributes:
| Attribute | Type | Description |
|-----------|------|-------------|
| `components_` | `np.ndarray (n_components, n_features)` | Principal axes (rows = eigenvectors) |
| `explained_variance_` | `np.ndarray (n_components,)` | Variance captured by each component |
| `explained_variance_ratio_` | `np.ndarray (n_components,)` | Fraction of total variance |
| `singular_values_` | `np.ndarray (n_components,)` | Singular values |
| `mean_` | `np.ndarray (n_features,)` | Per-feature mean used for centering |

**Example:**
```python
from ml_from_scratch.unsupervised import PCA
import numpy as np

X = np.random.randn(500, 100)

# Keep 95% of variance
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)

print(f"Components retained: {pca.n_components_}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Reconstruct
X_reconstructed = pca.inverse_transform(X_reduced)
reconstruction_error = np.mean((X - X_reconstructed)**2)
print(f"Reconstruction MSE: {reconstruction_error:.4f}")
```

---

## Neural

```python
from ml_from_scratch.neural import MLP
from ml_from_scratch.neural.activations import relu, sigmoid, softmax, tanh
from ml_from_scratch.neural.losses import mse, binary_cross_entropy, categorical_cross_entropy
```

---

### `MLP`

Multilayer perceptron with configurable architecture, activations, and optimizer.

```python
class MLP:
    def __init__(
        self,
        layers: list[int],                    # e.g. [784, 256, 128, 10]
        activation: str = "relu",             # hidden layer activation
        output_activation: str = "softmax",   # output activation
        weight_init: str = "he",              # "he" | "xavier" | "random"
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
    ) -> None
```

#### `fit`
```python
def fit(
    self,
    X: np.ndarray,              # (n_samples, n_features)
    y: np.ndarray,              # (n_samples,) integer labels or (n_samples, n_classes) one-hot
    optimizer: str = "adam",    # "sgd" | "momentum" | "rmsprop" | "adam"
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 32,
    val_split: float = 0.0,
    verbose: bool = True,
    callbacks: list | None = None,
) -> "MLP"
```

#### `predict`
```python
def predict(self, X: np.ndarray) -> np.ndarray
# Classification: argmax of output layer
# Regression: raw output
```

#### `predict_proba`
```python
def predict_proba(self, X: np.ndarray) -> np.ndarray
# Returns softmax probabilities, shape (n_samples, n_classes)
```

#### Attributes:
| Attribute | Type | Description |
|-----------|------|-------------|
| `weights_` | `list[np.ndarray]` | Weight matrices per layer |
| `biases_` | `list[np.ndarray]` | Bias vectors per layer |
| `loss_history_` | `dict` | `{"train": [...], "val": [...]}` |

**Example:**
```python
from ml_from_scratch.neural import MLP
from ml_from_scratch.utils import load_mnist, accuracy_score

X_train, y_train, X_test, y_test = load_mnist(normalize=True)

model = MLP(
    layers=[784, 512, 256, 10],
    activation="relu",
    output_activation="softmax",
    weight_init="he",
    dropout_rate=0.3,
)
model.fit(
    X_train, y_train,
    optimizer="adam",
    lr=1e-3,
    epochs=30,
    batch_size=256,
    val_split=0.1,
)
print("Test accuracy:", accuracy_score(y_test, model.predict(X_test)))
```

---

### Activations

All functions operate element-wise on `np.ndarray` and support batched inputs.

```python
from ml_from_scratch.neural.activations import relu, leaky_relu, sigmoid, tanh, softmax, elu

relu(x: np.ndarray) -> np.ndarray
    # f(x) = max(0, x). Derivative: (x > 0).astype(float)

leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray
    # f(x) = x if x > 0 else alpha * x

sigmoid(x: np.ndarray) -> np.ndarray
    # f(x) = 1/(1 + exp(-clip(x, -500, 500)))

tanh(x: np.ndarray) -> np.ndarray
    # f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

softmax(x: np.ndarray, axis: int = -1) -> np.ndarray
    # Numerically stable: subtracts max before exp

elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray
    # f(x) = x if x > 0 else alpha * (exp(x) - 1)
```

---

### Losses

```python
from ml_from_scratch.neural.losses import mse, mae, binary_cross_entropy, categorical_cross_entropy

mse(y_true: np.ndarray, y_pred: np.ndarray) -> float
    # Mean squared error. Gradient: (2/n) * (y_pred - y_true)

mae(y_true: np.ndarray, y_pred: np.ndarray) -> float
    # Mean absolute error. Gradient: sign(y_pred - y_true) / n

binary_cross_entropy(
    y_true: np.ndarray,  # (n,), binary labels
    y_pred: np.ndarray,  # (n,), sigmoid outputs
    eps: float = 1e-12,
) -> float

categorical_cross_entropy(
    y_true: np.ndarray,  # (n, K), one-hot
    y_pred: np.ndarray,  # (n, K), softmax outputs
    eps: float = 1e-12,
) -> float
```

---

## Utilities

```python
from ml_from_scratch.utils import (
    make_regression, make_classification, make_blobs,
    train_test_split, batch_iterator,
    accuracy_score, precision_recall_fscore, confusion_matrix,
    normalize, standardize, one_hot_encode,
)
```

### Data Generation

```python
make_regression(
    n_samples: int = 100,
    n_features: int = 10,
    n_informative: int = 5,   # features with nonzero coefficients
    noise: float = 0.1,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]  # X (n,p), y (n,)

make_classification(
    n_samples: int = 100,
    n_features: int = 10,
    n_classes: int = 2,
    n_informative: int = 5,
    class_sep: float = 1.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]  # X (n,p), y (n,) integer labels

make_blobs(
    n_samples: int = 300,
    n_features: int = 2,
    centers: int | np.ndarray = 3,
    cluster_std: float = 1.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]
```

### Metrics

```python
accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float

precision_recall_fscore(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",  # "macro" | "micro" | "binary"
) -> tuple[float, float, float]  # precision, recall, f1

confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray
# Returns: (n_classes, n_classes) integer matrix

r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float
# 1 - SS_res/SS_tot

mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float
```

### Preprocessing

```python
train_test_split(
    *arrays: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int | None = None,
) -> list[np.ndarray]  # alternating train/test for each array

standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]
# Returns: X_scaled, mean, std  (fit on X, use mean/std for test)

normalize(X: np.ndarray, norm: str = "l2") -> np.ndarray
# norm: "l1" | "l2" | "max"

one_hot_encode(y: np.ndarray, n_classes: int | None = None) -> np.ndarray
# Returns: (n_samples, n_classes)

batch_iterator(
    X: np.ndarray,
    y: np.ndarray | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
) -> Generator
```
