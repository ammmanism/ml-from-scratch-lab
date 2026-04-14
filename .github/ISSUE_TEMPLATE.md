---
name: 🐛 Bug Report
about: Create a report to help us improve the ML-from-Scratch core logic.
title: "[BUG] <Short Description>"
labels: bug, research-verification
assignees: ''

---

## 📋 Description
Provide a clear and concise description of the numerical discrepancy or engineering bug.

## 🛠️ Reproduction Steps
1. Dataset used (e.g., MNIST, Synthetic Circles):
2. Model parameters (layers, learning rate, regularization):
3. Code snippet:
   ```python
   # Your code here
   ```

## 📉 Expected vs. Actual Behavior
- **Expected**: (e.g., "Loss should decrease monotonically with Adam")
- **Actual**: (e.g., "Gratients exploded after 3 iterations at layer 2")

## 🧬 Environment Information
- **OS**: [e.g. Ubuntu 22.04]
- **Python Version**: [e.g. 3.9.1]
- **NumPy Version**: [e.g. 1.24.0]
- **Repository Branch**: [main/dev]

## ✅ Verification Checklist
- [ ] I have checked the `math_foundations.md` to ensure my intuition is correct.
- [ ] I have attempted a **Gradient Check** (finite differences) to isolate the error.
- [ ] I have searched the existing [Issues](https://github.com/ammmanism/ml-from-scratch-lab/issues).
