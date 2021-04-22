import numpy as np

import loupe


class Node:
    """Base class for nodes in the computational graph"""

    # When return type is ambiguous to Numpy, prefer loupe Node objects over
    # Numpy ndarrays
    __array_priority__ = 1

    def __add__(self, other):
        return loupe.numeric.add(self, other)

    def __sub__(self, other):
        return loupe.numeric.subtract(self, other)

    def __rsub__(self, other):
        return loupe.numeric.subtract(other, self)

    def __mul__(self, other):
        return loupe.numeric.multiply(self, other)

    def __pow__(self, exponent):
        return loupe.numeric.power(self, exponent)

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

    def __getitem__(self, index):
        return loupe.numeric.slice(self.data, index)

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
    inputs : list of :class:`~loupe.array`
        Array objects that the function operates on.

    Notes
    -----
    ``Function`` only defines the interface for objects that operate on 
    :class:`~loupe.array` objects - it doesn't actually implement any useful
    functionality. A complete list of available functions is available
    :ref:`here <api.functions>`.

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
        """Tuple of :attr:`data` dimensions.""" 
        return self._data.shape

    @property
    def inputs(self):
        """List of array inputs the function operates on."""
        return self._inputs

    @property
    def requires_grad(self):
        """Is ``True`` if any of the function's inputs have 
        ``requires_grad=True``, ``False`` otherwise.
        """
        return any([input.requires_grad for input in self.inputs])

    @property
    def cache(self):
        """Data that has been saved by :func:`cache_for_backward`.

        Returns
        -------
        cache : list

        """
        return self._cache

    def cache_for_backward(self, *data):
        """Save data for a future call to :func:`backward`.

        Saved data can be accessed through the :attr:`cache` attribute. 

        Parameters
        ----------
        data : list_like
            List of data to cache. Cached data should *probably* be ndarrays,
            but nothing prevents the caching of any Python type.

        Warning
        -------
        Care must be taken that cached data is not modified in place after
        calling :func:`forward`. Doing so will make the cached data stale and
        its resulting use during a call to :func:`backward` will be invalid.

        Notes
        -----
        This method should be called at most once, and only from inside the 
        :func:`forward` method.

        """
        self._cache = data

    def forward(self):
        """Performs the operation.

        This method should by overridden by all subclasses.

        Returns
        -------
        out : ndarray
            The result of the operation.

        """
        raise NotImplementedError
        
    def backward(self, grad):
        """Defines a formula for differentiating the operation.

        This method should be overridden by all subclasses.

        Parameters
        ----------
        grad : array_like
            The gradient with respect to the function output.

        """
        raise NotImplementedError
