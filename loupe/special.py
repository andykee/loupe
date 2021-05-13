import numpy as np

import loupe


class absolute_square(loupe.core.Function):
    """Calculate the absolute value squared, element-wise.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : :class:`~loupe.core.Function`
        Element-wise *abs(x**2)*

    Notes
    -----
    :class:`abs_square` is an alias for :class:`absolute_square`.
    
    """
    def __init__(self, x):
        self.input = loupe.asarray(x)
        super().__init__(self.input)

    def forward(self):
        input = self.input.getdata()
        result = np.square(np.abs(input))
        self.cache_for_backward(input)
        return result

    def backward(self, grad):
        input, = self.cache
        grad = 2 * input * grad
        self.input.backward(grad)


abs_square = absolute_square