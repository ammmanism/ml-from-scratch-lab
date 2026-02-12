import numpy as np

def zeros(shape: tuple) -> np.ndarray:
    """Initialize weights with zeros."""
    return np.zeros(shape)

def ones(shape: tuple) -> np.ndarray:
    """Initialize weights with ones."""
    return np.ones(shape)

def random_normal(shape: tuple, mean: float = 0.0, std: float = 0.01) -> np.ndarray:
    """Initialize weights with a random normal distribution."""
    return np.random.normal(mean, std, size=shape)

def random_uniform(shape: tuple, low: float = -0.01, high: float = 0.01) -> np.ndarray:
    """Initialize weights with a random uniform distribution."""
    return np.random.uniform(low, high, size=shape)

def xavier_uniform(shape: tuple) -> np.ndarray:
    """
    Xavier/Glorot uniform initialization.
    Suitable for Sigmoid/Tanh activations.
    
    Limit is sqrt(6 / (fan_in + fan_out))
    """
    fan_in, fan_out = _get_fans(shape)
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)

def xavier_normal(shape: tuple) -> np.ndarray:
    """
    Xavier/Glorot normal initialization.
    
    Std is sqrt(2 / (fan_in + fan_out))
    """
    fan_in, fan_out = _get_fans(shape)
    std = np.sqrt(2 / (fan_in + fan_out))
    return np.random.normal(0, std, size=shape)

def he_uniform(shape: tuple) -> np.ndarray:
    """
    He Kaiming uniform initialization.
    Suitable for ReLU activations.
    
    Limit is sqrt(6 / fan_in)
    """
    fan_in, _ = _get_fans(shape)
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)

def he_normal(shape: tuple) -> np.ndarray:
    """
    He Kaiming normal initialization.
    
    Std is sqrt(2 / fan_in)
    """
    fan_in, _ = _get_fans(shape)
    std = np.sqrt(2 / fan_in)
    return np.random.normal(0, std, size=shape)

def _get_fans(shape: tuple) -> tuple:
    """
    Helper to calculate fan_in and fan_out.
    Assumes shape is (n_inputs, n_outputs) or (n_outputs, n_inputs) depending on convention.
    Standard dense layer convention in many libs is (n_inputs, n_outputs).
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 1:
        fan_in = shape[0]
        fan_out = 1
    else:
        # CNNs etc.
        receptive_field_size = np.prod(shape[:-2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
        
    return fan_in, fan_out
