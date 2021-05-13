import numpy as np
import loupe


class tensordot(loupe.core.Function):
    """Compute tensor dot product along specified axes.

    Parameters
    ----------
    a, b : array_like
        Tensors to dot.
    axes : int or (2,) array_like, optional
        * If an int N, sum over the last N axes of *a* and the first N axes of
          *b* in order. The sizes of the corresponding axes must match.
        * If array_like, sum over the corresponding axes of *a* in the first 
          entry and the corresponding axes of *b* in the second entry. Both
          elements in axes must have the same length.

    Returns
    -------
    out : :class:`~loupe.core.Function`
        The tensor dot product of the input.

    See also
    --------
    :class:`~loupe.einsum`

    """
    def __init__(self, a, b, axes=2):
        self.a = loupe.asarray(a)
        self.b = loupe.asarray(b)
        self.axes = np.asarray(axes)

        self.a_axes = None
        self.b_axes = None
        self.c_axes = None

        super().__init__(a,b)

    def forward(self):
        a = self.a.getdata()
        b = self.b.getdata()
        self.cache_for_backward(a, b)

        result, a_axes, b_axes, c_axes = _tensordot(a, b, self.axes, self.a_axes, self.b_axes, self.c_axes)

        self.a_axes = a_axes
        self.b_axes = b_axes
        self.c_axes = c_axes

        return result

    def backward(self, grad):
        a, b = self.cache

        if self.a.requires_grad:
            a_grad, *_ = _tensordot(grad, b, axes=self.axes, 
                                    a_axes=self.c_axes,
                                    b_axes=[self.b_axes[1], self.b_axes[0]],
                                    c_axes=self.a_axes)
            self.a.backward(a_grad)

        if self.b.requires_grad:
            b_grad, *_ = _tensordot(a, grad, axes=self.axes, 
                                    a_axes=[self.a_axes[1], self.a_axes[0]],
                                    b_axes=self.c_axes,
                                    c_axes=self.b_axes)
            self.b.backward(b_grad)


def _tensordot(a, b, axes, a_axes, b_axes, c_axes):

    # chainer.tensordot.forward()
    if a_axes is None or b_axes is None:
        a_axes = [[], []]  # 0:row axes, 1:col axes
        b_axes = [[], []]  # 0:row axes, 1:col axes
        if axes.ndim != 0:
            a_axes[1], b_axes[0] = axes
            if np.isscalar(a_axes[1]):
                a_axes[1] = a_axes[1],
            if np.isscalar(b_axes[0]):
                b_axes[0] = b_axes[0],
        else:
            a_axes[1] = range(a.ndim - axes, a.ndim)
            b_axes[0] = range(axes)
        a_range = range(a.ndim)
        a_axes[0] = [i for i in a_range if i not in a_axes[1]]
        b_range = range(b.ndim)
        b_axes[1] = [i for i in b_range if i not in b_axes[0]]
    
    # chainer.tensordot._tensordot()
    a_col_ndim = len(a_axes[1])
    b_row_ndim = len(b_axes[0])
    if a_col_ndim != b_row_ndim:
        raise ValueError('axes count mismatch')
    if a.ndim < a_col_ndim or b.ndim < b_row_ndim:
        raise ValueError('dimension of input arrays must be '
                         'greater equal to dot-axes count '
                         f'({a_col_ndim})')
    for a_axis, b_axis in zip(a_axes[1], b_axes[0]):
        if a.shape[a_axis] != b.shape[b_axis]:
            raise ValueError('shape mismatch')

    y = np.tensordot(a, b, axes=(tuple(a_axes[1]), tuple(b_axes[0])))

    if c_axes is not None:
        a_row_ndim = len(a_axes[0])
        b_col_ndim = len(b_axes[1])
        c_row_ndim = len(c_axes[0])
        c_col_ndim = len(c_axes[1])
        if a_row_ndim != c_row_ndim:
            raise ValueError('axes count mismatch')
        if b_col_ndim != c_col_ndim:
            raise ValueError('axes count mismatch')

        trans = [None for i in range(y.ndim)]
        table_a = [1 if i in a_axes[0] else 0 for i in range(a.ndim)]
        table_a = np.cumsum(table_a) - 1
        for i, c_axis in enumerate(c_axes[0]):
            trans[c_axis] = table_a[a_axes[0][i]]
        table_b = [1 if i in b_axes[1] else 0 for i in range(b.ndim)]
        table_b = np.cumsum(table_b) - 1
        for i, c_axis in enumerate(c_axes[1]):
            trans[c_axis] = table_b[b_axes[1][i]] + len(a_axes[0])
        for i, c_axis in enumerate(trans):
            if i != c_axis:
                y = np.transpose(y, trans)
                break
    
    # chainer.tensordot.forward()
    if c_axes is None:
        c_axes = [[], []]  # 0:row axes, 1:col axes
        c_row_ndim = len(a_axes[0])
        c_col_ndim = len(b_axes[1])
        c_axes[0] = range(c_row_ndim)
        c_axes[1] = range(c_row_ndim, c_row_ndim + c_col_ndim)

    return y, a_axes, b_axes, c_axes