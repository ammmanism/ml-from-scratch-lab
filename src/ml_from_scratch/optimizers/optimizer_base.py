"""Abstract base class for all optimizers."""

from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Abstract base class for all optimizers."""
    
    @abstractmethod
    def update(self, params, grads):
        """Update parameters based on gradients."""
        pass