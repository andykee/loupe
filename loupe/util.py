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
        :class:`array` or :class:`~loupe.core.Function`, the original input is
        returned.
    
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
    centroid : tuple
        ``(x,y)`` centroid location.

    """
    img = np.asarray(img)
    img = img/np.sum(img)
    nr, nc = img.shape
    yy, xx = np.mgrid[0:nr, 0:nc]

    x = np.dot(xx.ravel(), img.ravel())
    y = np.dot(yy.ravel(), img.ravel())

    return x, y

def shift(a, shift):
    """Shift an array via FFT.

    Shift an array by (row, column). The shifts may be non-integer as the 
    shift operation is implemented by introducing a Fourier-domain tilt. If 
    ``a`` is complex, the result will also be complex.

    Parameters
    ----------
    a : array_like
        The input array.
    shift : (2,) sequence
        The shift specified as (row, column).
    Returns
    -------
    shifted : ndarray
        The shifted input array.

    Example
    -------
    .. code:: pycon

        >>> import loupe
        >>> import numpy as np
        >>> arr = np.zeros((3,3))
        >>> arr[2,2] = 1
        >>> arr
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 1.]])
        >>> arr_shift = loupe.shift(arr, shift=(-1,-1))
        >>> arr_shift
        array([[ 0.00000000e+00, -7.40148683e-17, -2.46716228e-17],
               [-1.16747372e-16,  1.00000000e+00,  2.14548192e-16],
               [-3.12823642e-17,  2.22044605e-16, -4.18468327e-17]])
    """
    a = np.asarray(a)

    dr, dc = shift
    R = dr * np.fft.fftfreq(a.shape[0])
    C = dc * np.fft.fftfreq(a.shape[1])

    RR, CC = np.meshgrid(R, C, indexing='ij')
    K = np.exp(-1j*2*np.pi*(RR+CC))
    shifted = np.fft.ifft2(np.fft.fft2(a)*K)

    if np.any(np.iscomplex(a)):
        return shifted
    else:
        return shifted.real


def register(arr, ref, oversample, return_error=False):
    """Compute the subpixel image translation to register the input array to a
    reference array.

    The registration shift is computed in two steps: first a coarse estimate 
    is computed from the FFT-based cross-correlation of the two input arrays.
    This estimate is then refined to subpixel accuracy by computing the 
    upsampled DFT-based cross-correlation in a small neigborhood around the
    initial estimate.

    Parameters
    ----------
    arr : array_like
        Array to register.
    ref : array_like
        Target array.
    oversample : float
        Oversampling factor for subpixel registration. Registration accuracy
        is 1/oversample.
    return_error : bool, optional
        If True, the noramlized RMS registration error is returned. Default is
        False.
    Returns
    -------
    shift : tuple
        Translation in (row, col) that will register *arr* to *ref*.
    err : float
        Registration error

    References
    ----------
    Guizar-Sicairos, Thurman, and Fienup, "Efficient subpixel image 
    registration algorithms". Optics Letters 33, 156-158 (2008)

    See also
    --------
    :func:`~loupe.shift`

    Example
    -------
    .. code:: pycon

        >>> import loupe
        >>> import numpy as np
        >>> ref = np.zeros((3,3))
        >>> ref[1,1] = 1
        >>> arr = np.zeros((3,3))
        >>> arr[2,2] = 1
        >>> shift = loupe.register(arr, ref, oversample=2)
        >>> shift
        (-1.0, -1.0)

    """
    F = np.fft.fft2(arr)
    G = np.fft.fft2(ref)
    xcorr = np.fft.fftshift(np.fft.ifft2(G*np.conj(F)))
    
    # find peak
    maxima = np.unravel_index(np.argmax(np.abs(xcorr)), xcorr.shape)
    peak = xcorr[maxima]

    # compute shifts
    center = np.array([np.fix(x/2) for x in arr.shape])
    shift = maxima - center
    if oversample !=1:
        # now we can set up and perform the oversampled dft on an oversampled 
        # 1.5 x 1.5 pixel region about the peak
        npix_dft = np.ceil(oversample*1.5)
        dft_shift = np.fix(npix_dft/2)
        rs = dft_shift - shift[0] * oversample
        cs = dft_shift - shift[1] * oversample

        # Compute DFT
        X = np.arange(arr.shape[1]) - np.floor(arr.shape[1]/2)
        Y = np.arange(arr.shape[0]) - np.floor(arr.shape[0]/2)
        U = np.arange(npix_dft) - cs
        V = np.arange(npix_dft) - rs
        E1 = np.exp(-2*np.pi*1j/(arr.shape[0]*oversample)*np.outer(V,Y))
        E2 = np.exp(-2*np.pi*1j/(arr.shape[1]*oversample)*np.outer(X,U))
        xcorr = np.dot(np.dot(E1,np.conj(np.fft.ifftshift(G*np.conj(F)))),E2)
        
        maxima_subpx = np.unravel_index(np.argmax(np.abs(xcorr)), xcorr.shape)
        peak = xcorr[maxima_subpx]

        # Combine subpixel peak coordinates with integer pixel peak coords
        maxima_subpx -= dft_shift 
        shift[0] += maxima_subpx[0]/oversample
        shift[1] += maxima_subpx[1]/oversample

    shift = tuple(shift)

    # Compute normalized RMS error
    if return_error:
        arr_amp = np.sum(np.abs(F)**2)
        ref_amp = np.sum(np.abs(G)**2)
        err = 1-np.abs(peak)**2/(arr_amp*ref_amp)
        err = np.sqrt(np.abs(err))
        return shift, err

    return shift