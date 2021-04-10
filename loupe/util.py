import numpy as np

import loupe

def asarray(a):
    if isinstance(a, (loupe.array, loupe.core.Function)):
        return a
    elif isinstance(a, (int, float, complex, list, tuple, np.ndarray)):
        return loupe.array(a)
    else:
        raise TypeError(f'Unsupported type. Cannot create array from \
                          {a.__class__.__name__}')


def zeros(shape, dtype=None, requires_grad=False):
    """Return a new array with values set to zeros."""
    return loupe.array(np.zeros(shape), dtype=dtype, 
                       requires_grad=requires_grad)


def zeros_like(a, dtype=None, requires_grad=False):
    """Return a new array of zeros with the same shape and type as a given 
    array.
    
    """
    return loupe.zeros(shape=a.shape, dtype=dtype, 
                       requires_grad=requires_grad)

def ones(shape, dtype=None, requires_grad=False):
    """Return a new array with values set to ones."""
    return loupe.array(np.ones(shape), dtype=dtype, 
                       requires_grad=requires_grad)


def ones_like(a, dtype=None, requires_grad=False):
    """Return a new array of ones with the same shape and type as a given 
    array.
    
    """
    return loupe.ones(shape=a.shape, dtype=dtype, 
                      requires_grad=requires_grad)


def rand(low=0.0, high=1.0, size=None, dtype=None, requires_grad=False):
    """Return a new array with values drawn from a uniform distribution."""
    return loupe.array(np.random.uniform(low=low, high=high, size=size), 
                 dtype=dtype, requires_grad=requires_grad)


def randn(loc=0.0, std=1.0, size=None, dtype=None, requires_grad=False):
    """Return a new array with values drawn from a normal distribution."""
    return loupe.array(np.random.normal(loc=loc, scale=std, size=size), dtype=dtype, 
                       requires_grad=requires_grad)