# 🛠️ Development & CI/CD Tooling

This directory contains the automation suite for the `pure-ml` engine. These scripts ensure architectural integrity, manage the pedagogical pipeline, and maintain a clean, high-performance development environment.

## 🚀 Core Automation Tools

### 1. `notebook_generator.py`
**Pedagogical Asset Manager** This tool programmatically generates, validates, and analyzes the `math_to_code/` curriculum. It ensures that the transition from LaTeX-based theory to Python-based implementation remains consistent across all 40+ notebooks.

* **Capabilities:** Programmatic cell injection, structural integrity validation, and curriculum statistics gathering.
* **Usage:**
    ```python
    from notebook_generator import NotebookGenerator
    gen = NotebookGenerator()
    # Programmatically audit the curriculum structure
    gen.analyze_notebook("math_to_code/01_linear_models/01_regression.ipynb")
    ```

### 2. `verify_notebooks.py`
**The CI/CD Gatekeeper** A rigorous validation utility that ensures every interactive notebook in the repository is executable and mathematically sound. This script is integrated into our GitHub Actions pipeline to prevent regression.

* **Usage:**
    ```bash
    # Validate JSON structure and metadata
    python scripts/verify_notebooks.py
    
    # Full execution test (Ensures convergence proofs still pass)
    python scripts/verify_notebooks.py --execute
    ```

### 3. `cleanup_scripts.py`
**Environmental Hygiene Utility** Maintains the "Pure" philosophy by purging temporary artifacts, localized caches, and checkpoint bloat.

* **Cleans:** `__pycache__`, `.pytest_cache`, `.ipynb_checkpoints`, and compiled `.pyc` artifacts.
* **Usage:**
    ```bash
    python scripts/cleanup_scripts.py
    ```

---

## 🏗️ Integration with CI/CD

These tools are not just standalone scripts; they are the backbone of our **GitHub Actions** workflows. 

| Workflow | Script Triggered | Purpose |
| :--- | :--- | :--- |
| **Pull Request Audit** | `verify_notebooks.py --execute` | Blocks merge if a notebook fails to converge. |
| **Staging Cleanup** | `cleanup_scripts.py` | Ensures artifacts aren't leaked into production builds. |

---

## 🧪 Requirements for Execution

While the `engine/` only requires NumPy, the development tooling requires the following for notebook execution and validation:

```bash
pip install nbformat nbconvert
