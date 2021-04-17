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
        # REF: https://numpy.org/doc/stable/user/basics.dispatch.html
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
    """Create a multidimensional array.
    
    Parameters
    ----------
    object : array_like
        An array or other data type that can be interpreted as an array.
    requires_grad : bool, optional
        It True, gradients will be computed for this array. Default is False.
    dtype : data-type, optional
        The desired data-type for the array. If not given, the type will be
        inferred from the supplied data object.
    """

    # TODO: masking, scaling, bounds?

    def __init__(self, object, requires_grad=False, dtype=None):

        self._data = np.asarray(object, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = np.zeros(self._data.shape, dtype=float)

    def __getitem__(self, index):
        # TODO: this action will break backward calculations
        # Need to develop a custom slice class?
        return self.data[index]

    def __setitem__(self, index, value):
        value = np.asarray(value, dtype=self.dtype)
        self._data[index] = value

    @property
    def data(self):
        """The array's data.

        Returns
        -------
        data : ndarray
        """
        return self._data

    @property
    def dtype(self):
        """Data type of the array's elements."""
        return self._data.dtype

    @property
    def shape(self):
        """Tuple of array dimensions."""
        return self._data.shape

    @property
    def requires_grad(self):
        """Is ``True`` if gradients need to be computed for this 
        :class:`array`, ``False`` otherwise.
        """
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._requires_grad = value

    @property
    def grad(self):
        """Gradient of the array. 
        
        This attribute is zero by default and is only computed when 
        :attr:`requires_grad` is True and 
        :func:`~loupe.Function.backward` is called.

        Returns
        -------
        grad : ndarray

        """
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    def zero_grad(self):
        """Zero the array gradient."""
        self.grad = np.zeros_like(self._data)

    def flatten(self):
        """Return the array flattened into one dimension."""
        return self._data.flatten()

    def backward(self, grad):
        """Compute the gradient of the array.

        Parameters
        ----------
        grad : array_like
            Gradient with respect to the array.

        """
        if not self.requires_grad:
            pass
        else:
            # accumulate gradient
            if self.dtype in (np.float32, np.float64):
                self.grad += grad.real
            else:
                self.grad += grad


class Function(Node):
    """Base class for representing functions that operate on :class:`~loupe.array` 
    objects.

    Parameters
    ----------
    inputs : list of :class:`array`
        Array objects that the function operates on.

    """
    def __init__(self, *inputs):
        self._inputs = inputs
        self._cache = []
        self._data = self.forward()

    @property
    def data(self):
        """The result of evaluating the function.

        Returns
        -------
        data : ndarray
        """
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
