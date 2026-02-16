"""Datasets module initialization."""

from .synthetic_data import make_regression, make_classification
from .data_utils import train_test_split

__all__ = ['make_regression', 'make_classification', 'train_test_split']