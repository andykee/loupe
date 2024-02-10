import numpy as np
from numpy.lib.stride_tricks import as_strided

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


class rebin(loupe.core.Function):
    """Rebin an array.  

    Parameters
    ----------
    x : array_like
        Input array
    bin_shape: tuple
        The shape of the bin. Each dimension must divide evenly into the
        corresponsing dimension of `x`.
    
    Returns
    -------
    out : :class:`~loupe.core.Function`
        Rebinned array
    """

    def __init__(self, x, bin_shape):
        self.input = loupe.asarray(x)
        self.bin_shape =  np.broadcast_to(bin_shape, (2,))
        
        if (self.bin_shape <= 0).any():
            raise ValueError("'bin_shape' elements must be positive")
        
        if (self.input.shape % self.bin_shape).sum() != 0:
            raise ValueError("'bin_shape' is not compatible with input array shape")

        super().__init__(self.input)

    def forward(self):
        input = self.input.getdata()

        # the code below is adapted from the skimage function view_as_blocks()
        new_shape = tuple(input.shape // self.bin_shape) + tuple(self.bin_shape)
        new_strides = tuple(input.strides * self.bin_shape) + input.strides
        restrided_input = as_strided(input, shape=new_shape, strides=new_strides)

        result = np.einsum('ijkl->ij', restrided_input)
        
        self.cache_for_backward(input.shape, result.shape)

        return result

    def backward(self, grad):
        input_shape, result_shape, = self.cache

        if result_shape != grad.shape:
            raise ValueError(f'grad must have shape = {result_shape}')
        
        new_shape = tuple(grad.shape) + tuple(self.bin_shape)
        new_strides = tuple(grad.strides) + (0, 0)
        restrided_grad = as_strided(grad, shape=new_shape, strides=new_strides)
        grad = np.copy(np.swapaxes(restrided_grad, 1, 2)).reshape(input_shape)

        self.input.backward(grad)


class normavg(loupe.core.Function):

    def __init__(self, x):
        self.input = loupe.asarray(x)
        super().__init__(self.input)

    def forward(self):
        input = self.input.getdata()
        mean = np.sum(input)
        result = np.reciprocal(mean) * input
        self.cache_for_backward(mean)
        return result

    def backward(self, grad):
        print('grad pre:', np.mean(grad))
        mean, = self.cache
        grad = np.reciprocal(mean) * grad
        #grad += -1*np.reciprocal(mean)
        print('grad_post:', np.mean(grad))
        self.input.backward(grad)


