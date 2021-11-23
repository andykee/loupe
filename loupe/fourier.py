import functools

import numpy as np
from numpy.lib.stride_tricks import as_strided

import loupe


class fft2(loupe.core.Function):
    def __init__(self, input):
        self.input = loupe.asarray(input)
        super().__init__(self.input)

    def forward(self):
        return np.fft.fft2(self.input.data)

    def backward(self, grad):
        return np.fft.ifft2(grad)


class ifft2(loupe.core.Function):
    def __init__(self, input):
        self.input = loupe.asarray(input)
        super().__init__(self.input)

    def forward(self):
        return np.fft.ifft2(self.input.data)

    def backward(self, grad):
        return np.fft.fft2(grad)


class dft2(loupe.core.Function):
    """Compute the 2-dimensional discrete Fourier Transform.

    This function allows independent control over input shape, output shape,
    and output sampling by implementing the matrix triple product algorithm
    described in [1].

    Parameters
    ----------
    x : array_like
        Input array.
    alpha : (2,) array_like
        Output plane sampling frequency in terms of (row, col).
    shape : (2,) arrqy_like, optional
        Shape of the output. If `shape` is None (default), the shape of the input
        is used.
    unitary : bool, optional
        Normalization flag. If True, a normalization is performed on the output
        such that the DFT operation is unitary. Default is True.
    shift : (2,) array_like, optional
        Number of pixels in (r,c) to shift the DC pixel in the output plane with
        the origin centrally located in the plane. Default is (0, 0).

    Returns
    -------
    out : :class:`~loupe.core.Function`
        The discrete Fourier transform of the input.

    Notes
    -----
    * `dft2` expects the DC pixel to be centered in the input array.
      The center is assumed to be at ``np.floor(n/2) + 1`` for length `n`.
      Note this is consistent with a shifted FFT performed as:
      ``F = ifftshift(fft2(fftshift(f)))``

    * Performing the DFT with ``alpha = 1/f.shape``, ``unitary = False``,
      and ``shape = f.shape`` is equivalent to a shifted FFT:

    .. code:: pycon

        >>> f = loupe.rand(size=(10, 10))
        >>> F = loupe.dft2(f, alpha=(1/10, 1/10), unitary=False, shape=f.shape)
        >>> G = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(f)))
        >>> np.allclose(F, G)
        True

    References
    ----------
    [1] Soummer, et. al. Fast computation of Lyot-style coronagraph propagation (2007)

    """
    def __init__(self, x, alpha, shape=None, unitary=True, shift=(0,0)):

        self.input = loupe.asarray(x)
        self.alpha = loupe.asarray(alpha)
        self.output_shape = shape
        self.unitary = unitary
        self.shift = shift
        super().__init__(self.input, self.alpha)

    def forward(self):
        input = self.input.getdata()
        alpha = _sanitize_ordered_pair(self.alpha.getdata(), dtype=float)
        shift = _sanitize_ordered_pair(self.shift, dtype=float)

        shape, strides = _as_strided_args(input, self.output_shape)
        result = np.empty(shape, dtype=complex)
        W_row, W_row_conj, W_col, W_col_conj = [], [], [], []

        for n, x in enumerate(as_strided(input, shape=shape, strides=strides, writeable=False)):
            Wr = _dft2_matrix(x.shape[0], shape[1], alpha[0], shift[0])
            Wc = _dft2_matrix(x.shape[1], shape[2], alpha[1], shift[1])
            tmp = np.dot(Wr.T, x)
            result[n] = np.dot(tmp, Wc)

            W_row.append(Wr)
            W_row_conj.append(np.conj(Wr))
            W_col.append(Wc)
            W_col_conj.append(np.conj(Wc))

        result = np.squeeze(result)

        if self.unitary:
            root_alpha = np.sqrt(np.abs(alpha))
            norm_coeff = np.prod(root_alpha)
        else:
            root_alpha = 1
            norm_coeff = 1

        # NOTE: result is purposely saved before we apply unitary normalization.
        # The result that is used in the backward function is expected to be
        # "clean" (i.e. unitary=False)
        self.cache_for_backward(input, alpha, W_row, W_row_conj, W_col, W_col_conj,
                                result, root_alpha, norm_coeff)

        if self.unitary:
            result *= norm_coeff

        return result

    def backward(self, grad):

        # unpack saved_inputs
        (input, alpha, W_row, W_row_conj, W_col, W_col_conj,
         result, root_alpha, norm_coeff) = self.cache

        # backward() needs the "clean" (non-unitary) gradient. If unitary = True
        # we need to multiply the incoming gradient by norm_coeff. Because
        # norm_coeff is set to 1 in forward() if unitary = False, it is safe to
        # always multiply grad by norm_coeff here.
        grad_clean = grad * norm_coeff

        shape, strides = _as_strided_args(grad_clean, input.shape[-2:])
        input_grad = np.empty(shape, dtype=complex)

        for n, x in enumerate(as_strided(grad_clean, shape=shape, strides=strides, writeable=False)):
            h_grad = np.dot(W_row_conj[n], x)
            input_grad[n] = np.dot(h_grad, W_col_conj[n].T)

        input_grad = np.squeeze(input_grad)

        self.input.backward(input_grad)


@functools.lru_cache(maxsize=32)
def _dft2_matrix(n, N, alpha, shift):
    w = -2.0 * np.pi * np.outer(_coords(n)-shift, _coords(N)-shift)
    E = np.exp(1j * w * alpha)
    return E


@functools.lru_cache(maxsize=32)
def _coords(n):
    #TODO: ensure n is a python primitive type
    # It needs to be hashable for LRU cache to work
    return np.arange(n) - np.floor(n/2.0)


def _sanitize_ordered_pair(x, dtype):
    x = np.asarray(x)

    if x.shape != (2,):
            raise ValueError(f"can't interpret x with shape {x.shape} as ordered pair")

    return [dtype(x[0]), dtype(x[1])]


def _as_strided_args(input, output_shape):
    depth = 1 if input.ndim ==2 else input.shape[0]

    shape = output_shape if output_shape is not None else input.shape[-2:]
    if len(input.shape) == 2:
        shape = (1, *shape)
    else:
        shape = (input.shape[0], *shape)

    strides = input.strides if depth > 1 else (0, *input.strides)
    print(shape, strides)
    return shape, strides
