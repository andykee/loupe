import numpy as np
import loupe

def test_absolute_square():
    a = loupe.rand(low=-1, high=1, size=(10,10))
    res = loupe.absolute_square(a)
    assert np.array_equal(res, np.abs(a)**2)

def test_absolute_square_backward():
    a = loupe.rand(low=-1, high=1, size=(10,10), requires_grad=True)
    res = loupe.absolute_square(a)
    res.backward(np.ones((10,10)))
    assert np.array_equal(a.grad, 2*a)