# 🔬 Research Experiments & Benchmarking

The `experiments/` directory is the "Laboratory" of this repository. Here, we conduct scientific validation of our implementations against industry baselines and explore model behaviors across various regimes.

---

## 🏗️ 1. Scientific Verification Suites

### A. Gradient Checking
Every gradient implementation in the `neural/` module is validated for numerical correctness.
- **Run**: `pytest tests/test_gradients.py`
- **Logic**: We use the central difference formula:
  $$\frac{\partial J}{\partial \theta} \approx \frac{J(\theta + \epsilon) - J(\theta - \epsilon)}{2\epsilon}$$
- **Pass Status**: Relative error must be $< 10^{-7}$.

### B. Convergence Comparison
We compare the convergence rates of different optimizers (SGD, Momentum, Adam) on standard benchmarks like MNIST.
- refer to `benchmarks/compare_sklearn.ipynb` for the automated comparison harness.

---

## ⚙️ 2. Hyperparameter Sweeps
We utilize the `configs/` directory to manage and launch experiments without modifying source code.

### Running a Sweep
To test a model architecture with different hidden layer sizes:
1. Modify `configs/model_configs.yaml`.
2. Launch the generator:
   ```bash
   python scripts/notebook_generator.py --config configs/model_configs.yaml --output experiments/mlp_sweep.ipynb
   ```
3. Execute the resulting notebook to visualize the performance frontier.

---

## 📊 3. Baseline Validation (vs. Scikit-Learn)
We maintain a strict **Accuracy Parity** goal. Our models should achieve within $1\%$ of the accuracy of `sklearn` on the same datasets with the same hyperparameters.

| Experiment | Target | Status |
| :--- | :--- | :--- |
| **LR on Iris** | $> 95\%$ Accuracy | ✅ |
| **MLP on MNIST** | $> 98\%$ Accuracy | ✅ |
| **Lasso Sparsity** | $> 80\%$ Zero Weights | ✅ |

---

## 🛠️ 4. Adding a New Experiment
1. Create a data-generation config in `configs/synthetic_datasets.yaml`.
2. Define your model architecture in `configs/model_configs.yaml`.
3. Use `scripts/verify_notebooks.py` to ensure your experimental notebook remains reproducible.