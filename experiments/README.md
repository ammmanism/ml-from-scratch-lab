# Experiments

> Reproducible ablation studies, convergence analysis, hyperparameter sweeps, and sklearn benchmarks.

---

## Table of Contents

- [Running Benchmarks](#1-running-benchmarks)
- [Hyperparameter Sweeps](#2-hyperparameter-sweeps)
- [Ablation Studies](#3-ablation-studies)
- [Gradient Validation](#4-gradient-validation)
- [Convergence Analysis](#5-convergence-analysis)
- [Reproducing README Plots](#6-reproducing-readme-plots)

---

## 1. Running Benchmarks

Compare our implementations against scikit-learn on accuracy, speed, and memory.

### Quick run (all models):
```bash
python benchmarks/compare_sklearn.py --model all --verbose
```

### Specific model:
```bash
python benchmarks/compare_sklearn.py \
    --model linear_regression \
    --dataset boston \
    --n-samples 1000 \
    --n-features 20
```

### Output format:
```
Model                    Dataset         sklearn    ours      ratio   acc_diff
LinearRegression         boston(506,13)  0.002s     0.003s    0.67x   +0.001
LogisticRegression       digits(1797,64) 0.12s      0.18s     0.67x   -0.002
DecisionTree (depth=5)   wine(178,13)    0.003s     0.009s    0.33x   0.000
RandomForest (n=50)      wine(178,13)    0.08s      0.15s     0.53x   -0.005
KMeans (k=3)             iris(150,4)     0.011s     0.019s    0.58x   ARI diff: 0.00
```

### Save results to JSON:
```bash
python benchmarks/compare_sklearn.py --output results/benchmark_$(date +%Y%m%d).json
```

---

## 2. Hyperparameter Sweeps

Use YAML configs to define sweep spaces:

**`configs/training_config.yaml`:**
```yaml
model: LogisticRegression
dataset: digits
sweep:
  lr: [0.001, 0.01, 0.1, 1.0]
  epochs: [100, 500, 1000]
  regularization: [null, "l1", "l2"]
  C: [0.01, 0.1, 1.0, 10.0]
metric: accuracy
n_seeds: 3
```

**Run sweep:**
```bash
python scripts/run_sweep.py --config configs/training_config.yaml --output results/sweep.json
```

**Visualize results:**
```bash
python scripts/plot_sweep.py --results results/sweep.json
```

This generates:
- Heatmap of accuracy vs (lr, C)
- Best hyperparameter combination
- Statistical summary across seeds

---

## 3. Ablation Studies

Measure the contribution of individual components.

**MLP ablation (dropout, batch norm, initialization):**
```bash
python experiments/ablation_mlp.py \
    --dataset mnist \
    --ablate dropout batch_norm weight_init \
    --epochs 20 \
    --output results/mlp_ablation.json
```

**Sample output:**
```
Component removed      Test Acc    Delta vs Full
─────────────────────────────────────────────
Full model             0.971       —
Remove dropout         0.958       -0.013
Remove batch norm      0.965       -0.006
Random init (no He)    0.941       -0.030
Remove both reg.       0.934       -0.037
```

**Decision tree depth ablation:**
```bash
python experiments/ablation_tree_depth.py \
    --max-depths 1 2 3 5 10 20 None \
    --dataset wine
```

---

## 4. Gradient Validation

Validate that your analytical gradients match numerical finite-difference approximations.

### Run all gradient checks:
```bash
pytest tests/test_gradients.py -v --timeout=120
```

### Validate a specific backward pass:
```bash
python scripts/check_gradient.py --model MLP --layers 4 8 3 --batch-size 5
```

**Manual gradient check:**
```python
from scipy.optimize import check_grad
import numpy as np

def validate_gradient(model, X, y, param_name="weights"):
    """Check gradient of model loss at current params."""

    def loss(flat_params):
        model._load_flat_params(flat_params)
        return model._compute_loss(X, y)

    def grad(flat_params):
        model._load_flat_params(flat_params)
        model._forward(X)
        model._backward(y)
        return model._flat_grads()

    params_init = model._flat_params()
    error = check_grad(loss, grad, params_init)
    print(f"Gradient check error ({param_name}): {error:.2e}")
    assert error < 1e-5, f"FAILED: gradient error too large"
    print("PASSED ✓")
```

---

## 5. Convergence Analysis

Compare optimizers on the same problem.

```bash
python experiments/convergence_analysis.py \
    --model MLP \
    --optimizers sgd momentum adam rmsprop \
    --dataset mnist \
    --epochs 50 \
    --lr-per-optimizer 0.01 0.01 0.001 0.001 \
    --output results/convergence.json
```

This produces:
- Loss vs epoch curves for all optimizers
- Final accuracy comparison table
- Wall-clock time comparison

**Reproduce the README convergence plot:**
```bash
python experiments/convergence_analysis.py \
    --preset readme_figure \
    --save-plot docs/assets/convergence_plot.png
```

---

## 6. Reproducing README Plots

All plots in the README are reproducible:

```bash
# Benchmark table (README Performance section)
python benchmarks/compare_sklearn.py --model all --format table

# Convergence chart
python experiments/convergence_analysis.py --preset readme_figure

# Memory pie chart (MNIST MLP)
python experiments/memory_profile.py --model MLP --layers 784 512 256 10 --dataset mnist

# Generate all at once
make figures
```

---

## Experiment Tracking

We use simple JSON logging — no MLflow/W&B dependency.

```python
from ml_from_scratch.utils import ExperimentLogger

with ExperimentLogger("results/my_experiment.json") as log:
    log.config(model="LinearRegression", lr=0.01, epochs=1000)
    model.fit(X_train, y_train)
    log.metric("train_mse", train_mse)
    log.metric("test_mse", test_mse)
    log.metric("fit_time_s", elapsed)
```

Results are stored as line-delimited JSON, readable with pandas:
```python
import pandas as pd
df = pd.read_json("results/my_experiment.json", lines=True)
```
