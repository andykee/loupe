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
            self.left.backward(grad)

        if self.right.requires_grad:
            self.right.backward(grad)

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
            self.left.backward(grad)

        if self.right.requires_grad:
            self.right.backward(-grad) 

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
            self.left.backward(left_grad)
        
        if self.right.requires_grad:
            right_grad = np.conj(left) * grad
            self.right.backward(right_grad)


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