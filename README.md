
# 🚀 ML From Scratch Lab

**Implementing Machine Learning from First Principles with NumPy**

A research-grade educational repository that builds machine learning algorithms from scratch using only NumPy and Python. Dive deep into the mathematical foundations, implement gradient-based optimization, construct neural networks layer by layer, and validate against industry-standard libraries like scikit-learn. Designed for researchers, engineers, and students who believe that true understanding comes from building.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-green.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/yourusername/ml-from-scratch-lab/pulls)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/ml-from-scratch-lab)

---

## 🔭 Project Vision

**Why build from scratch?**  
In an era of high-level APIs and auto-ML, the fundamental mechanisms of machine learning are often obscured. This lab exists to peel back the layers—to implement every algorithm, every optimization step, and every backpropagation pass with transparent, readable code.

**Educational & Research Goals**  
- Provide a **self-contained curriculum** that progresses from linear algebra foundations to transformer architectures.  
- Serve as a **reference implementation** for researchers prototyping new ideas without heavy framework dependencies.  
- Demonstrate **engineering best practices** in a research context: modular design, comprehensive testing, and reproducible benchmarks.

**Engineering Philosophy**  
- **Mathematical Transparency**: Every equation in a paper should map directly to a line of code.  
- **Performance Awareness**: Write efficient NumPy code without sacrificing clarity.  
- **Educational First**: Code is documentation; notebooks are tutorials.

---

## ✨ Feature Highlights

- **🧮 Pure NumPy Implementations**  
  No black boxes—everything from linear regression to multi-head attention is built with `np.dot`, `np.einsum`, and manual gradients.

- **📉 Gradient-Based Optimization**  
  Implement SGD, Momentum, Adam, and more with explicit gradient computation and optional autodiff for educational clarity.

- **🧠 Deep Learning Foundations**  
  Modular layers (Dense, Conv1D, RNN, LSTM), activation functions, loss functions, and backpropagation through time.

- **🔍 Transformer Basics**  
  Scaled dot-product attention, positional encoding, encoder/decoder blocks—all from scratch, ready for experimentation.

- **🔬 Research Experiments**  
  Easily swap components, log metrics, and compare convergence against reference libraries.

- **⚡ Benchmarking vs. sklearn**  
  Automated tests assert numerical correctness and performance parity with scikit-learn’s reference implementations.

---

## 📂 Repository Architecture

```
ml-from-scratch-lab/
├── notebooks/                      # Interactive learning path
│   ├── 00_mathematical_foundations
│   ├── 01_linear_models
│   ├── 02_tree_models
│   ├── 03_neural_networks
│   ├── 04_transformers
│   └── 05_experiments
├── src/ml_from_scratch/            # Core library
│   ├── core/                       # Autodiff engine, parameter handling
│   ├── optimizers/                  # SGD, Adam, etc.
│   ├── models/                      # High-level model APIs
│   ├── neural/                       # Layers, activations, losses
│   ├── datasets/                     # Synthetic data generators
│   └── utils/                        # Metrics, preprocessing
├── tests/                           # Unit tests & integration tests
├── benchmarks/                       # Performance & correctness benchmarks
├── examples/                         # Standalone usage scripts
├── configs/                          # Experiment configuration files
├── docs/                             # Sphinx documentation
└── scripts/                          # Utility scripts (data download, etc.)
```

---

## 📘 Notebook Learning Path

Follow the notebooks in order to build understanding from the ground up:

| Step | Topic                      | Key Concepts                                      |
|------|----------------------------|---------------------------------------------------|
| 00   | **Mathematical Foundations** | Linear algebra, calculus, probability – all with NumPy |
| 01   | **Linear Models**            | Linear regression, logistic regression, gradient descent |
| 02   | **Tree Models**              | Decision trees, random forests, gradient boosting  |
| 03   | **Neural Networks**          | MLPs, backpropagation, activations, regularisation |
| 04   | **Transformers**             | Attention, positional encoding, transformer blocks |
| 05   | **Experiments**              | Hyperparameter tuning, ablation studies            |

Each notebook is a self-contained lesson with visualizations, mathematical derivations, and code exercises.

---

## ⚙️ Installation

### Using `pip` (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-from-scratch-lab.git
cd ml-from-scratch-lab

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install the package in editable mode
pip install -e .
```

### Dependencies

- Python 3.9+
- NumPy >= 1.24
- Matplotlib (for visualizations)
- Jupyter (to run notebooks)
- scikit-learn (for benchmarks)

Install all development dependencies:

```bash
pip install -r requirements-dev.txt
```

---

## 🚀 Quick Start Examples

### Linear Regression from Scratch

```python
from ml_from_scratch.models import LinearRegression
from ml_from_scratch.datasets import make_regression
from ml_from_scratch.utils import train_test_split

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train, lr=0.01, epochs=1000)

# Predict
predictions = model.predict(X_test)
print("MSE:", ((predictions - y_test) ** 2).mean())
```

### Multilayer Perceptron (MLP)

```python
from ml_from_scratch.neural import Model, Dense, ReLU, Softmax
from ml_from_scratch.optimizers import Adam
from ml_from_scratch.losses import CrossEntropy

# Build a simple network
model = Model()
model.add(Dense(128, input_dim=784))
model.add(ReLU())
model.add(Dense(10))
model.add(Softmax())

# Compile with optimizer and loss
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=CrossEntropy())

# Train on MNIST-like data (X_train, y_train)
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

### Running Benchmarks

```bash
python benchmarks/compare_logistic_regression.py
```

---

## 🧠 Engineering Principles

- **🔬 Reproducibility** – Every experiment seeds random generators and logs hyperparameters. Configuration files ensure results can be replicated.
- **📐 Mathematical Transparency** – Code mirrors equations. For example, the gradient of MSE is implemented exactly as `2 * (y_pred - y_true) / n_samples`.
- **⚡ Performance Awareness** – We vectorize operations, avoid Python loops, and use NumPy’s optimized routines. Benchmark scripts compare runtime with scikit-learn.
- **🎓 Research Orientation** – The codebase is modular so you can swap components (e.g., replace an activation function) and observe the effect.

---

## 📊 Benchmarks & Validation

All core models are tested against scikit-learn’s implementations using identical data and hyperparameters. The `benchmarks/` directory contains scripts that:

- Assert numerical correctness (e.g., weights, predictions) within a tight tolerance.
- Compare training time and convergence curves.
- Generate reports showing parity.

We also maintain a suite of unit tests (`tests/`) that cover edge cases, gradient checks, and model serialization.

---

## 🗺️ Roadmap

- [ ] **LLM from Scratch** – Build a small GPT-style transformer with byte-pair encoding.
- [ ] **Diffusion Models** – Implement denoising diffusion probabilistic models.
- [ ] **Reinforcement Learning** – Basic algorithms (DQN, PPO) using only NumPy.
- [ ] **GPU Acceleration** – Optional CuPy backend for large-scale experiments.
- [ ] **Differentiable Programming** – Add a minimal autodiff engine to replace manual gradients.

Contributions and ideas are always welcome!

---

## 🤝 Contributing

We follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct. To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Write tests for your changes.
4. Ensure code passes `black`, `flake8`, and `pytest`.
5. Open a pull request with a clear description of the problem and solution.

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

**Author**: [AMMAN HUSSAIN ANSARI](https://github.com/ammmanism)  
