import numpy as np

import loupe


class Node:
    """Base class for nodes in the computational graph"""

    # When return type is ambiguous to Numpy, prefer loupe Node objects over
    # Numpy ndarrays
    __array_priority__ = 1

    def __add__(self, other):
        return loupe.math.add(self, other)

    def __sub__(self, other):
        return loupe.math.subtract(self, other)

    def __rsub__(self, other):
        return loupe.math.subtract(other, self)

    def __mul__(self, other):
        return loupe.math.multiply(self, other)

    def __pow__(self, exponent):
        return loupe.math.power(self, exponent)

    __radd__ = __add__
    __rmul__ = __mul__

    def __array__(self):
        # https://numpy.org/doc/stable/user/basics.dispatch.html
        return self.data
    
    def __str__(self):
        return np.array_str(self.data)

    def __repr__(self):
        prefix = 'array('
        suffix = ')'
        array_str = np.array2string(self.data, prefix=prefix, suffix=suffix)
        return prefix + array_str + suffix

    @property
    def data(self):
        return None

    @property
    def requires_grad(self):
        return False


class array(Node):
    """Multidimensional array"""

    def __init__(self, object, requires_grad=False, dtype=None):

        self._data = np.asarray(object, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = np.zeros(self._data.shape, dtype=float)

    def __getitem__(self, index):
        # TODO: this action will break backward calculations
        # Need to develop a custom slice class?
        return self.data[index]

    def __setitem__(self, index, value):
        try:
            value = np.asarray(value, dtype=self.dtype)
            self._data[index] = value
        except IndexError as e:
            raise e

    @property
    def data(self):
        return self._data

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

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

    def backward(self, grad):
        if not self.requires_grad:
            pass
        else:
            # accumulate gradient
            if self.dtype in (np.float32, np.float64):
                self.grad += grad.real
            else:
                self.grad += grad


class Function(Node):
    """Base class for all functions"""
    def __init__(self, *inputs):
        self._inputs = inputs
        self._cache = []
        self._data = self.forward()

    @property
    def data(self):
        self._data = np.asarray(self.forward())
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def inputs(self):
        return self._inputs

    @property
    def requires_grad(self):
        return any([input.requires_grad for input in self.inputs])

    @property
    def cache(self):
        return self._cache

    def cache_for_backward(self, *data):
        self._cache = data

    def forward(self):
        raise NotImplementedError
        
    def backward(self, grad):
        raise NotImplementedError


def asarray(a):
    if isinstance(a, (array, Function)):
        return a
    elif isinstance(a, (int, float, complex, list, tuple, np.ndarray)):
        return array(a)
    else:
        raise TypeError(f'Unsupported type. Cannot create array from \
                          {a.__class__.__name__}')


def rand(low=0.0, high=1.0, size=None, dtype=None, requires_grad=False):
    """Return a new array with values drawn from a uniform distribution."""
    return array(np.random.uniform(low=low, high=high, size=size), 
                 dtype=dtype, requires_grad=requires_grad)