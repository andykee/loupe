import numpy as np

import loupe

class add(loupe.core.Function):
    """Add two arrays"""

    def __init__(self, x1, x2):
        self.left = loupe.asarray(x1)
        self.right = loupe.asarray(x2)
        super().__init__(self.left, self.right)

    def forward(self):
        left = self.left()
        right = self.right()
        result = np.add(left, right)
        return result
    
    def backward(self, grad):
        left = self.left()
        right = self.right()

        if self.left.requires_grad:
            left_grad = _broadcast_grad(grad, left.shape)
            self.left.backward(left_grad)

        if self.right.requires_grad:
            right_grad = _broadcast_grad(grad, right.shape)
            self.right.backward(right_grad)

class subtract(loupe.core.Function):
    """Subtract two arrays"""

    def __init__(self, x1, x2):
        self.left = loupe.asarray(x1)
        self.right = loupe.asarray(x2)
        super().__init__(self.left, self.right)

    def forward(self):
        left = self.left()
        right = self.right()
        result = np.subtract(left, right)
        return result
    
    def backward(self, grad):
        
        left = self.left()
        right = self.right()

        if self.left.requires_grad:
            left_grad = _broadcast_grad(grad, left.shape)
            self.left.backward(left_grad)

        if self.right.requires_grad:
            right_grad = _broadcast_grad(-grad, right.shape)
            self.right.backward(right_grad) 

class multiply(loupe.core.Function):
    """Multiply two arraays"""

    def __init__(self, x1, x2):
        self.left = loupe.asarray(x1)
        self.right = loupe.asarray(x2)
        super().__init__(self.left, self.right)

    def forward(self):
        left = self.left()
        right = self.right()
        result = np.multiply(left, right)
        return result 

    def backward(self, grad):
    
        left = self.left()
        right = self.right()

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
    """Raise array to a power"""
    def __init__(self, x1, x2):
        self.input = loupe.asarray(x1)
        self.exp = x2
        super().__init__(self.input)

    def forward(self):
        input = self.input()
        return np.power(input, self.exp)

    def backward(self, grad):
        input = self.input()
        grad = self.exp * np.power(input, self.exp-1) * grad
        self.input.backward(grad)