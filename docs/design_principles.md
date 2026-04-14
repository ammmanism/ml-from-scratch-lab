# Design Principles of ML-From-Scratch-Lab

## 1. Zero "Black Box" Philosophy
The primary goal of this repository is education. State-of-the-art libraries like PyTorch, TensorFlow, and Scikit-Learn provide incredible abstraction that allows engineers to solve real-world problems rapidly. However, this abstraction hides the complex mathematical reality of how models learn. This project exists to pull back the curtain.

## 2. NumPy-Only Core
To truly understand matrix multiplication, gradient flow, and backpropagation, we enforce a strict **NumPy-only** rule for all core arithmetic logic. 
- No `torch.backward()` or automatic differentiation algorithms. The chain rule is computed and hardcoded explicitly for each layer and activation.
- `scikit-learn` is used strictly in `tests/` and `benchmarks/` to validate our models against an industry standard.

## 3. Scikit-Learn Consistent API
All models expose `fit(X, y)` and `predict(X)` or `predict_proba(X)`. By maintaining API consistency with `sklearn`, it is easier for learners to swap our custom classifiers seamlessly into the workflows they are already familiar with.

## 4. Code Over Comments
We prefer readable code with explicit mathematical variable names (e.g., `Z`, `A`, `dW`, `db`) that map 1:1 with the accompanying Jupyter Notebook derivations. Mathematical consistency trumps PEP8 naming if PEP8 obfuscates the formula.
