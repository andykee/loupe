import numpy as np


class array:
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