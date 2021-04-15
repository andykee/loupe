import numpy as np

import loupe

class add(loupe.core.Function):
    """Add arrays element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be added. Arrays must either have the same shape or be
        broadcastable to a common shape (which becomes the shape of the
        output).

    Returns
    -------
    out : Function
        The sum of *x1* and *x2*, element-wise.

    Notes
    -----
    Equivalent to *x1 + x2*.

    Examples
    --------

    .. code:: pycon

        >>> loupe.add(1.0, 4.0)
        5.0
        >>> x1 = loupe.array([[1, 2], [3, 4]])
        >>> x2 = loupe.array([1, 2])
        >>> loupe.add(x1, x2)
        array([[ 2.,  4.],
               [ 4.,  6.]])
    
    The ``+`` operator can be used as shorthand for ``loupe.add``.

    .. code:: pycon

        >>> x1 = loupe.array([[1, 2], [3, 4]])
        >>> x2 = loupe.array([1, 2])
        >>> x1 + x2
        array([[ 2.,  4.],
               [ 4.,  6.]])

    """
    def __init__(self, x1, x2):
        self.left = loupe.asarray(x1)
        self.right = loupe.asarray(x2)
        super().__init__(self.left, self.right)

    def forward(self):
        left = self.left.data
        right = self.right.data
        self.cache_for_backward(left, right)
        
        result = np.add(left, right)
        return result
    
    def backward(self, grad):
        left, right = self.cache

        if self.left.requires_grad:
            left_grad = _broadcast_grad(grad, left.shape)
            self.left.backward(left_grad)

        if self.right.requires_grad:
            right_grad = _broadcast_grad(grad, right.shape)
            self.right.backward(right_grad)

class subtract(loupe.core.Function):
    """Subtract arrays element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be subtracted from each other. Arrays must either have 
        the same shape or be broadcastable to a common shape (which becomes 
        the shape of the output).

    Returns
    -------
    out : Function
        The difference of *x1* and *x2*, element-wise.

    Notes
    -----
    Equivalent to *x1 - x2*.

    Examples
    --------

    .. code:: pycon

        >>> loupe.subtract(1.0, 4.0)
        -3.0
        >>> a = loupe.array([[1, 2], [3, 4]])
        >>> b = loupe.array([1, 2])
        >>> loupe.subtract(a, b)
        array([[ 0.,  0.],
               [ 2.,  2.]])
    
    The ``-`` operator can be used as shorthand for ``loupe.subtract``.

    .. code:: pycon

        >>> a = loupe.array([[1, 2], [3, 4]])
        >>> b = loupe.array([1, 2])
        >>> a - b
        array([[ 0.,  0.],
               [ 2.,  2.]])

    """
    def __init__(self, x1, x2):
        self.left = loupe.asarray(x1)
        self.right = loupe.asarray(x2)
        super().__init__(self.left, self.right)

    def forward(self):
        left = self.left.data
        right = self.right.data
        self.cache_for_backward(left, right)

        result = np.subtract(left, right)
        return result
    
    def backward(self, grad):
        
        left, right = self.cache

        if self.left.requires_grad:
            left_grad = _broadcast_grad(grad, left.shape)
            self.left.backward(left_grad)

        if self.right.requires_grad:
            right_grad = _broadcast_grad(-grad, right.shape)
            self.right.backward(right_grad) 

class multiply(loupe.core.Function):
    """Multiply arrays element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be multiplied. Arrays must either have the same shape or 
        be broadcastable to a common shape (which becomes the shape of the
        output).

    Returns
    -------
    out : Function
        The product of *x1* and *x2*, element-wise.

    Notes
    -----
    Equivalent to *x1 * x2*.

    Examples
    --------

    .. code:: pycon

        >>> loupe.multiply(2.0, 4.0)
        8.0
        >>> x1 = loupe.array([[1, 2], [3, 4]])
        >>> x2 = loupe.array([1, 2])
        >>> loupe.multiply(x1, x2)
        array([[ 1.,  4.],
               [ 3.,  8.]])
    
    The ``*`` operator can be used as shorthand for ``loupe.multiply``.

    .. code:: pycon

        >>> x1 = loupe.array([[1, 2], [3, 4]])
        >>> x2 = loupe.array([1, 2])
        >>> x1 * x2
        array([[ 1.,  4.],
               [ 3.,  8.]])

    """
    def __init__(self, x1, x2):
        self.left = loupe.asarray(x1)
        self.right = loupe.asarray(x2)
        super().__init__(self.left, self.right)

    def forward(self):
        left = self.left.data
        right = self.right.data
        self.cache_for_backward(left, right)

        result = np.multiply(left, right)
        return result 

    def backward(self, grad):
    
        left, right = self.cache

        if self.left.requires_grad:
            left_grad = np.conj(right) * grad
            left_grad = _broadcast_grad(left_grad, left.shape)
            self.left.backward(left_grad)
        
        if self.right.requires_grad:
            right_grad = np.conj(left) * grad
            right_grad = _broadcast_grad(right_grad, right.shape)
            self.right.backward(right_grad)


def _broadcast_grad(grad, array_shape):
    # REF: https://github.com/joelgrus/autograd
    
    # sum out the added dimensions
    ndims_added = grad.ndim - len(array_shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)

    # sum across broadcasted (but not added) dimensions
    for n, dim in enumerate(array_shape):
        if dim == 1:
            grad = grad.sum(axis=n, keepdims=True)
    
    return grad


class power(loupe.core.Function):
    """First array elements raised to powers from second array, element-wise.

    Raise each base in *x1* to the positionally-corresponding power in *x2*. 
    *x1* and *x2* must be broadcastable to the same shape.

    Parameters
    ----------
    x1 : array_like
        Base array.
    x2 : array_like
        Exponent array.

    Returns
    -------
    out : Function
        The bases in *x1* raised to the exponent(s) in *x2*.

    Notes
    -----
    Equivalent to `x1 ** x2`.

    Examples
    --------
    Cube each element in an array.

    .. code:: pycon

        >>> x1 = loupe.array([1,2,3,4,5])
        >>> loupe.power(x1, 3)
        array([ 1., 8., 27., 64., 125.])

    Rase the bases to different exponents.

    .. code:: pycon

        >>> x2 = loupe.array([1,2,3,2,1])
        >>> loupe.power(x1, x2)
        array([ 1., 4., 27., 8., 5.]) 

    The `**` operator can be used as a shorthand for ``loupe.power``.

    .. code:: pycon

        >>> x = loupe.array([1,4,9,16])
        >>> x ** 0.5
        loupe.array([ 1., 2., 3., 4.])

    """
    def __init__(self, x1, x2):
        self.input = loupe.asarray(x1)
        self.exp = x2
        super().__init__(self.input)

    def forward(self):
        input = self.input.data
        self.cache_for_backward(input)

        return np.power(input, self.exp)

    def backward(self, grad):
        input, = self.cache
        
        grad = self.exp * np.power(input, self.exp-1) * grad
        self.input.backward(grad)