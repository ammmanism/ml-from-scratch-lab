# Scripts Directory

Utility scripts for managing the ML from Scratch project.

⚠️ **AI-Generated Code Notice**
All scripts in this directory have been generated or modified using AI assistance.
Please review and test thoroughly before using in production.

---

## Available Scripts

### 1. `notebook_generator.py`
**Unified Jupyter Notebook generator and manager**

Creates, validates, and analyzes Jupyter notebooks programmatically without hardcoded paths.

**Features:**
- Create code and markdown cells
- Save/load notebooks
- Analyze notebook structure
- Validate notebook integrity
- Get notebook statistics

**Usage:**
```python
from notebook_generator import NotebookGenerator

gen = NotebookGenerator()

# Create cells
cells = [
    gen.create_markdown_cell("# My Notebook"),
    gen.create_code_cell("import numpy as np")
]

# Create and save notebook
nb = gen.create_notebook(cells)
gen.save_notebook(nb, "output.ipynb")

# Analyze
gen.analyze_notebook("output.ipynb")

# Validate
gen.validate_notebook("output.ipynb")
```

---

### 2. `cleanup_scripts.py`
**Repository cleanup utility**

Removes temporary files, caches, and generated files to keep the repository clean.

**Cleans:**
- Generated notebook scripts (`gen_nb_*.py`)
- Python cache (`__pycache__/`)
- Pytest cache (`.pytest_cache/`)
- Compiled Python files (`.pyc`, `.pyo`)
- Jupyter notebook checkpoints

**Usage:**
```bash
python cleanup_scripts.py
```

**Features:**
- Comprehensive logging
- Glob pattern support
- Error handling and reporting
- Statistics summary

---

### 3. `verify_notebooks.py`
**Notebook verification and validation utility**

Validates notebook structure and optionally executes all notebooks to check for errors.

**Validates:**
- JSON structure
- Required keys (`cells`, `metadata`)
- Cell format
- (Optional) Code execution

**Usage:**
```bash
# Validate structure only
python verify_notebooks.py

# Validate structure and execute
python verify_notebooks.py --execute
# or
python verify_notebooks.py -x
```

**Features:**
- Structure validation (works without nbconvert)
- Optional execution with nbconvert
- Detailed error reporting
- Summary statistics

---

## Installation

Most dependencies are already included in the main environment. For execution features:

```bash
pip install nbformat nbconvert
```

---

## Running Scripts

### From project root:
```bash
python scripts/notebook_generator.py
python scripts/cleanup_scripts.py
python scripts/verify_notebooks.py
```

### From scripts directory:
```bash
cd scripts
python notebook_generator.py
python cleanup_scripts.py
python verify_notebooks.py
```

---

## Integration with CI/CD

These scripts can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Verify Notebooks
  run: python scripts/verify_notebooks.py --execute

- name: Cleanup Repository
  run: python scripts/cleanup_scripts.py
```

---

## Contributing

When adding new scripts:
1. Add AI-generated code notice at the top
2. Include comprehensive docstrings
3. Add logging for visibility
4. Test thoroughly
5. Document usage in this README

---

## License

See main repository LICENSE file.
