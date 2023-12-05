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
        Input array, can be complex. Must 
    alpha : (2,) array_like
        Output plane sampling frequency in terms of (row, col).
    shape : (2,) arrqy_like, optional
        Shape of the output. If `shape` is None (default), the shape of the input
        is used.
    shift : sequence of floats, optional
        Number of pixels in (r,c) to shift the DC pixel in the output plane with
        the origin centrally located in the plane. Default is (0, 0). Can be 
        2-dimensional to provide independent shifts for each Fourier transform
        if `x` is 3-dimensional.
    offset : sequence of floats, optional
        Offset of the center of the input `x` in terms of number of pixels in 
        (r,c). Default is (0, 0). Can be 2-dimensional to provide independent
        offsets for each Fourier transform if `x` is 3-dimensional.
    unitary : bool, optional
        Normalization flag. If True, a normalization is performed on the output
        such that the DFT operation is unitary. Default is True.
    axes : sequence of ints, optional
        Axes over which to compute the Fourier transform. If not given, the last 
        two axes are used.

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
    def __init__(self, x, alpha, shape=None, shift=(0,0), offset=(0,0), unitary=True, axes=None):

        self.input = loupe.asarray(x)
        if self.input.ndim not in (2,3):
            raise ValueError("Input array must have ndim == 2 or 3")

        self.alpha = loupe.asarray(alpha)
        self.out_shape, self.axes = _cook_nd_args(self.input, shape, axes)

        # figure out which index is the iteration index
        mask = np.ones(3, dtype=bool)
        mask[self.axes] = False
        self.iter_axis = np.squeeze(np.arange(3)[mask])

        # broadcast shift and offset as necessary
        depth = self.input.shape[self.iter_axis]

        self.shift = np.broadcast_to(shift, (depth, 2))
        self.offset = np.broadcast_to(offset, (depth, 2))
        self.unitary = unitary
        super().__init__(self.input, self.alpha)

    def forward(self):
        input = self.input.getdata()

        if input.ndim == 2:
            input = input[np.newaxis,:]

        alpha = np.broadcast_to(self.alpha.getdata(), (2,))

        # rearrange input so that it is arranged as
        # [iteration axis, rows, cols]
        input = np.moveaxis(input, [self.iter_axis, *self.axes], np.arange(3))
        
        result = []
        W_row, W_row_conj, W_col, W_col_conj = [], [], [], []

        for x, shift, offset in zip(input, self.shift, self.offset):
            Wr, Wc = _dft2_matrices(input.shape[1], input.shape[2],
                                    self.out_shape[0], self.out_shape[1],
                                    alpha[0], alpha[1],
                                    shift[0], shift[1],
                                    offset[0], offset[1])
            result.append(np.dot(Wr.dot(x), Wc))
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

        if grad_clean.ndim == 2:
            grad_clean = grad_clean[np.newaxis,:]

        input_grad = []
        for n, x in enumerate(grad_clean):
            Wr, Wc = W_row_conj[n], W_col_conj[n]
            input_grad.append(np.dot(Wr.T.dot(x), Wc.T))
        input_grad = np.squeeze(input_grad)

        self.input.backward(input_grad)


def _cook_nd_args(a, s=None, axes=None):
    # slightly modified version of numpy's function of the
    # same name
    if s is None:
        if axes is None:
            if a.ndim == 2:
                s = list(a.shape)
            elif a.ndim == 3:
                s = list(a.shape[1:3])
            else:
                raise ValueError("Array must have ndim == 2 or 3")
        else:
            s = np.take(a.shape, axes)
    s = list(s)
    if axes is None:
        axes = list(range(-len(s), 0))
    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    return s, axes


def _dft2_matrices(m, n, M, N, alphar, alphac, shiftr, shiftc, offsetr, offsetc):
    R, S, U, V = _dft2_coords(m, n, M, N)
    E1 = np.exp(-2.0 * 1j * np.pi * alphar * np.outer(R-shiftr+offsetr, U-shiftr)).T
    E2 = np.exp(-2.0 * 1j * np.pi * alphac * np.outer(S-shiftc+offsetc, V-shiftc))
    return E1, E2

@functools.lru_cache(maxsize=32)
def _dft2_coords(m, n, M, N):
    # R and S are (r,c) coordinates in the (m x n) input plane f
    # V and U are (r,c) coordinates in the (M x N) output plane F

    R = np.arange(m) - np.floor(m/2.0)
    S = np.arange(n) - np.floor(n/2.0)
    U = np.arange(M) - np.floor(M/2.0)
    V = np.arange(N) - np.floor(N/2.0)

    return R, S, U, V
