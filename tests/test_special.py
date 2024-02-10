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

def test_rebin():
    a = loupe.ones((10,10))
    res = loupe.rebin(a, (5,5))
    assert np.array_equal(res, [[25,25],[25,25]])

def test_rebin_backward():
    a = loupe.ones((4,4), requires_grad=True)
    res = loupe.rebin(a, (2,2))
    res.backward(np.array([[1,2],[3,4]]))
    assert np.array_equal(a.grad, [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]])