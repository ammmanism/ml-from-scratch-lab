# 🏗️ Design Principles: The Architecture of Understanding

This repository is built as a **Pedagogical Monument**. Every design decision is a tradeoff between readability, mathematical transparency, and numerical performance.

---

## 🧘 The Zen of ML From Scratch

1. **Explicit is better than Implicit**: We write the full gradient code rather than using autodiff frameworks.
2. **Readability is a Mathematical Requirement**: If a line of NumPy code doesn't map to a line of a research paper, it needs refactoring.
3. **No Black Boxes**: If an algorithm requires a random seed, a tolerance, or an epsilon, it must be an exposed parameter.
4. **Pedagogy > Performance**: While we vectorize operations for speed, we never sacrifice clarity for a 5% speed gain.
5. **The Tensor is Truth**: Understanding the shape of the data at every layer is 90% of the work.

---

## 🛠️ Architectural Tradeoffs

### 1. Why Pure NumPy?
High-level frameworks (PyTorch/Scikit-Learn) are tools for **production**, but they are "opaque" for **education**.
- **The Test**: If you can't implement Backpropagation using only `np.dot` and the Chain Rule, you don't yet understand how it works. 
- **The Reward**: Implementing from scratch forces you to confront numerical stability, broadcasting nuances, and memory management.

### 2. Standardized API Interface
We strictly adhere to the **Scikit-Learn API Pattern** (`fit`, `predict`, `predict_proba`).
- **Reasoning**: This allows our "pure" models to be tested directly against industry-standard benchmarks. 
- **Benefit**: You can swap `sklearn.linear_model.LogisticRegression` for `ml_from_scratch.linear.LogisticRegression` in any pipeline to verify correctness.

### 3. Modular Layer Blocks
In our `neural` module, we treat layers as standalone mathematical units.
- Each layer has a `forward` and `backward` method.
- This decoupling allows for the construction of arbitrary computational graphs (CNNs, MLPs, Transformers) using the same core logic.

---

## ⚡ Performance vs. Transparency
We utilize **Vectorized NumPy** operations over Python `for` loops. 
- **The Distinction**: A `for` loop across 60,000 images is slow. A matrix operation across a 60,000-row tensor is fast (C-backend). 
- **Principle**: We use vectorization because it mirrors the **Tensor Algebra** found in the literature, which actually makes the code *more* readable, not less.

---

## 🚀 Vision
To build a library that serves as the "Rosetta Stone" for machine learning—translating the complex notation of research papers into the clear, executable reality of Python.
