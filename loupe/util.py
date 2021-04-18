import numpy as np

import loupe

def asarray(a):
    """Convert the input to an array.
    
    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array. This 
        includes lists, lists of tuples, tuples, tuples of tuples, 
        tuples of lists and ndarrays.

    Returns
    -------
    out : :class:`array`
        Input data packaged as an :class:`array`. If input is already an
        :class:`array`, the original input is returned.
    
    Examples
    --------
    Convert a list into an array:

    .. code:: pycon

        >>> a = [1,2,3]
        >>> loupe.asarray(a)
        array([ 1.,2., 3.])

    Existing arrays are not copied:

    .. code:: pycon

        >>> a = loupe.array([1,2,3])
        >>> loupe.asarray(a) is a
        True

    """
    if isinstance(a, (loupe.array, loupe.core.Function)):
        return a
    elif isinstance(a, (int, float, complex, list, tuple, np.ndarray)):
        return loupe.array(a)
    else:
        raise TypeError(f'Unsupported type. Cannot create array from \
                          {a.__class__.__name__}')


def zeros(shape, dtype=None, requires_grad=False):
    """Return a new array with values set to zeros.
    
    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the array
    dtype : data type, optional
        Desired data type for the array
    requires_grad : bool, optional
        It True, gradients will be computed for this array. Default is False.

    Returns
    -------
    out : :class:`array`
        Array of zeros with the given shape and dtype.

    See Also 
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    ones : Return a new array with values set to ones.
    
    """
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


def centroid(img):
    """Compute image centroid location.

    Parameters
    ----------
    img : array_like
        Input array.

    Returns
    -------
    tuple
        ``(x,y)`` centroid location.
    """

    img = np.asarray(img)
    img = img/np.sum(img)
    nr, nc = img.shape
    yy, xx = np.mgrid[0:nr, 0:nc]

    x = np.dot(xx.ravel(), img.ravel())
    y = np.dot(yy.ravel(), img.ravel())

    return x, y                       