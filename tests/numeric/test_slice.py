import numpy as np
import loupe

def test_slice():
    a = loupe.rand(size=(10,10))
    res = loupe.slice(a, (slice(0,5), slice(0,5)))
    assert np.array_equal(res, a.data[0:5,0:5])

def test_slice_overloaded():
    a = loupe.rand(size=(10,10))
    res = a[0:5,0:5]
    assert np.array_equal(res, a.data[0:5,0:5])

def test_slice_backward():
    a = loupe.rand(size=(10,10), requires_grad=True)
    res = loupe.slice(a, (slice(0,5), slice(0,5)))
    res.backward(grad=np.ones((5,5)))

    result_grad = np.zeros_like(a)
    result_grad[0:5,0:5] = 1

    assert np.array_equal(a.grad, result_grad) 

def test_slice_overload_backward():
    a = loupe.rand(size=(10,10), requires_grad=True)
    res = a[0:5,0:5]
    res.backward(grad=np.ones((5,5)))

    result_grad = np.zeros_like(a)
    result_grad[0:5,0:5] = 1

    assert np.array_equal(a.grad, result_grad) 