# 🤝 Contributing: Maintaining the Elite Standard

This is a research-grade repository. To maintain the "GOAT-tier" engineering quality, all contributions must adhere to the high standards of production machine learning engineering.

---

## 🏗️ 1. Development Environment Setup
We use a strictly controlled environment to ensure reproducibility.

```bash
# Clone and setup
git clone https://github.com/ammmanism/ml-from-scratch-lab.git
cd ml-from-scratch-lab

# Initialize virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (Mandatory)
pre-commit install
```

---

## 🧪 2. The Verification Workflow
Before submitting a Pull Request, your code must pass the **Triple Verification** process.

### A. Mathematical Verification (Gradient Checking)
If you are adding a new neural layer or cost function, you **must** provide a unit test that uses the Finite Difference method to check your gradients.
- Refer to `tests/test_gradients.py` for the reference implementation.

### B. Static Analysis
We enforce PEP 8 and strong typing.
- `make lint` will run `black`, `flake8`, and `mypy`.
- Code with `Any` types or linting violations will be rejected by CI.

### C. Performance Regression
Compare your implementation against `sklearn`. A performance gap > 3x (speed) or 1% (accuracy) requires an architectural explanation in the PR.

---

## ✍️ 3. Documentation Excellence
We use **Google-Style Docstrings**.

```python
def fit(self, X: np.ndarray, y: np.ndarray) -> 'Model':
    """Fits the model to the training data.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target vector of shape (n_samples,).

    Returns:
        Self: Returns the instance itself.
        
    Raises:
        ValueError: If X and y have dimension mismatch.
    """
```

---

## 🚀 4. Submission Checklist
- [ ] Added pedagogical Notebook with LaTeX derivations.
- [ ] Added unit tests under `tests/`.
- [ ] Passed `make lint` and `make test`.
- [ ] Updated `docs/api_reference.md` if new public methods were added.
- [ ] Verified that all new Jupyter Notebooks execute top-to-bottom without error.

---

## ⚖️ Code of Conduct
This project is dedicated to professional learning. We prioritize technical clarity, mutual respect, and rigorous scientific debate. Bullying or non-constructive criticism is strictly prohibited.
