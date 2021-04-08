import numpy as np

import loupe

class Node:
    """Base class for nodes in the computational graph"""

    def __add__(self, other):
        return loupe.math.add(self, other)

    def __sub__(self, other):
        return loupe.math.subtract(self, other)

    def __rsub__(self, other):
        return loupe.math.subtract(other, self)

    def __mul__(self, other):
        return loupe.math.multiply(self, other)

    __radd__ = __add__
    __rmul__ = __mul__

    @property
    def requires_grad(self):
        return False


class array(Node):
    """Multidimensional array"""

    def __init__(self, object, requires_grad=False, dtype=None):

        self._data = np.asarray(object, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = np.zeros(self.shape, dtype=float)

    @property
    def data(self):
        return self._data

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._requires_grad = value

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value


class Function(Node):
    """Base class for all functions"""
    def __init__(self, *inputs):
        self._inputs = inputs

    def __call__(self):
        return np.asarray(self.forward())

    @property
    def inputs(self):
        return self._inputs

    @property
    def requires_grad(self):
        return any([input.requires_grad for input in self.inputs])

    def forward(self):
        raise NotImplementedError
        
    def backward(self, grad):
        raise NotImplementedError


def asarray(a):
    if isinstance(a, (array, Function)):
        return a
    elif isinstance(a, (int, float, complex, list, tuple, np.ndarray)):
        return loupe.array(a)
    else:
        raise TypeError(f'Unsupported type. Cannot create array from {a.__class__.__name__}')