import numpy as np
import loupe

def test_multiply():
    a = loupe.rand(size=(10,10))
    b = loupe.rand(size=(10,10))
    res = loupe.multiply(a,b)
    assert np.array_equal(res, np.multiply(a.data, b.data))

def test_multiply_overloaded():
    a = loupe.rand(size=(10,10))
    b = loupe.rand(size=(10,10))
    res = a * b
    assert np.array_equal(res, np.multiply(a.data, b.data))

def test_scalar_multiply():
    a = loupe.rand(size=(10,10))
    b = loupe.rand()
    res = a * b
    assert np.array_equal(res, np.multiply(a.data, b.data)) 

def test_multiply_backward():
    a = loupe.rand(size=(10,10), requires_grad=True)
    b = loupe.rand(size=(10,10), requires_grad=True)
    res = loupe.multiply(a,b)
    res.backward(grad=np.ones((10,10)))
    assert np.array_equal(a.grad, np.conj(b.data)) 
    assert np.array_equal(b.grad, np.conj(a.data)) 

def test_scalar_multiply_backward():
    a = loupe.rand(size=(10,10), requires_grad=True)
    b = loupe.rand(requires_grad=True)
    res = loupe.multiply(a,b)
    res.backward(grad=np.ones((10,10)))
    assert np.array_equal(a.grad, np.broadcast_to(np.conj(b.data), a.shape)) 
    assert np.allclose(b.grad, np.sum(np.conj(a.data)))

def test_broadcast_multiply():
    a = loupe.rand(size=(2,3))
    b = loupe.rand(size=3)
    res = a * b
    assert np.array_equal(res, np.multiply(a.data, b.data))

def test_broadcast_multiply2():
    a = loupe.rand(size=(2,3))
    b = loupe.rand(size=(1,3))
    res = a * b
    assert np.array_equal(res, np.multiply(a.data, b.data)) 

def test_broadcast_multiply_backward():
    a = loupe.rand(size=(2,3), requires_grad=True)
    b = loupe.rand(size=3, requires_grad=True)
    res = a * b 
    res.backward(grad=np.ones((2,3)))
    assert np.array_equal(a.grad, np.broadcast_to(np.conj(b.data), a.shape))
    assert np.array_equal(b.grad, np.sum(np.conj(a.data), 0))

def test_broadcast_multiply_backward2():
    a = loupe.rand(size=(2,3), requires_grad=True)
    b = loupe.rand(size=(1,3), requires_grad=True)
    res = a * b 
    res.backward(grad=np.ones((2,3)))
    assert np.array_equal(a.grad, np.broadcast_to(np.conj(b.data), a.shape))
    assert np.array_equal(b.grad, np.sum(np.conj([a.data]), 1))
