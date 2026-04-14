# 📘 Technical API Reference: The Definitive Spec

This document provides the formal technical specifications for the `ml-from-scratch-lib`. Each entry includes mathematical signatures, architectural notes, and complexity analysis.

---

## 🏗️ 1. Linear Submodule

### `LinearRegression`
Standard Ordinary Least Squares solver with ElasticNet $(L1/L2)$ regularization support.

- **Signature**: `LinearRegression(method: str = 'normal', alpha: float = 0.0, l1_ratio: float = 0.0)`
- **Mathematical Op**: $\min_\theta ||y - X\theta||_2^2 + \lambda ||\theta||_1 + \lambda ||\theta||_2^2$
- **Time Complexity**: 
  - `method='normal'`: $O(n^2 m + n^3)$ due to matrix inversion.
  - `method='gradient'`: $O(k \cdot n \cdot m)$ per $k$ iterations.
- **Usage Example**:
  ```python
  from ml_from_scratch.linear import LinearRegression
  model = LinearRegression(method='gradient', alpha=0.01)
  model.fit(X_train, y_train)
  y_hat = model.predict(X_test)
  ```

---

## 🧠 2. Neural Submodule

### `MLPClassifier`
Fully-connected modular Neural Network optimized for multi-class objectives.

- **Signature**: `MLPClassifier(hidden_layers: List[int], activations: List[str], optimizer: str = 'adam')`
- **Key Methods**:
  - `fit(X, y)`: Executes the Forward-Backward loop with Gradient Clipping.
  - `predict_proba(X)`: Returns Softmax normalized probabilities.
- **Architectural Note**: Uses "He Normal" initialization to maintain signal variance $\text{Var}(a) = \frac{2}{n_{in}}$.
- **Usage Example**:
  ```python
  from ml_from_scratch.neural import MLPClassifier
  # Elite Deep Arch: 256 -> 128 -> 10
  clf = MLPClassifier(hidden_layers=[256, 128], activations=['relu', 'relu'])
  clf.fit(X, y)
  ```

---

## 🌲 3. Trees Submodule

### `DecisionTreeClassifier`
Recursive CART implementation utilizing Gini Impurity for optimal greedy splitting.

- **Signature**: `DecisionTreeClassifier(max_depth: Optional[int] = None, min_samples_split: int = 2)`
- **Criterion**: $\text{Gini} = 1 - \sum_{i=1}^G p_i^2$
- **Efficiency Note**: Pre-sorts features to achieve $O(n \cdot m \log m)$ complexity during training.
- **Complexity**: $O(m^2 \cdot n)$ average case for tree growth.

---

## 🔍 4. Unsupervised Submodule

### `KMeans`
Centroid-based clustering with optimized Voronoi iteration.

- **Signature**: `KMeans(n_clusters: int, init: str = 'k-means++')`
- **Mechanism**: Minimizes Within-Cluster Sum of Squares (WCSS).
- **Optimization**: The `k-means++` initialization ensures probability of selecting center $c_i$ is proportional to $D(x)^2$, leading to $O(\log k)$ competitive solutions to the global optimum.

### `PCA`
- **Logic**: SVD-based projection.
- **Usage**: `pca = PCA(n_components=2); X_reduced = pca.fit_transform(X)`

---

## 🧮 5. Utility Layer

### `gradient_check`
- **Logic**: Compares analytical gradient $\nabla J$ with numerical approximation $\frac{J(\theta + \epsilon) - J(\theta - \epsilon)}{2\epsilon}$.
- **Accuracy**: Should yield a relative difference $< 10^{-7}$.
