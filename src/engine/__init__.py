"""
ML From Scratch Library

A pure NumPy implementation of machine learning algorithms with sklearn-like API.
"""

from . import utils
from .linear.linear_regression import LinearRegression
from .linear.logistic_regression import LogisticRegression
from .math.linear_algebra import gaussian_elimination, power_iteration
from .math.probability import sample_gaussian, simulate_clt
from .linear.regularization import l1_penalty, l2_penalty, elasticnet_penalty

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "gaussian_elimination",
    "power_iteration",
    "sample_gaussian",
    "simulate_clt",
    "l1_penalty",
    "l2_penalty",
    "elasticnet_penalty",
    "utils"
]