import numpy as np
import loupe

def test_add():
    a = loupe.rand(size=(10,10))
    b = loupe.rand(size=(10,10))
    res = loupe.add(a,b)
    assert np.array_equal(res, np.add(a.data, b.data))

def test_add_overloaded():
    a = loupe.rand(size=(10,10))
    b = loupe.rand(size=(10,10))
    res = a + b
    assert np.array_equal(res, np.add(a.data, b.data))

def test_scalar_add():
    a = loupe.rand(size=(10,10))
    b = loupe.rand()
    res = a + b
    assert np.array_equal(res, np.add(a.data, b.data)) 

def test_add_backward():
    a = loupe.rand(size=(10,10), requires_grad=True)
    b = loupe.rand(size=(10,10), requires_grad=True)
    res = loupe.add(a,b)
    res.backward(grad=np.ones((10,10)))
    assert np.array_equal(a.grad, np.ones_like(a.data)) 
    assert np.array_equal(b.grad, np.ones_like(b.data)) 

def test_scalar_add_backward():
    a = loupe.rand(size=(10,10), requires_grad=True)
    b = loupe.rand(requires_grad=True)
    res = loupe.add(a,b)
    res.backward(grad=np.ones((10,10)))
    assert np.array_equal(a.grad, np.ones_like(a.data)) 
    assert np.array_equal(b.grad, 100) 

def test_broadcast_add():
    a = loupe.rand(size=(2,3))
    b = loupe.rand(size=3)
    res = a + b
    assert np.array_equal(res, np.add(a.data, b.data))

def test_broadcast_add2():
    a = loupe.rand(size=(2,3))
    b = loupe.rand(size=(1,3))
    res = a + b
    assert np.array_equal(res, np.add(a.data, b.data)) 

def test_broadcast_add_backward():
    a = loupe.rand(size=(2,3), requires_grad=True)
    b = loupe.rand(size=3, requires_grad=True)
    res = a + b 
    res.backward(grad=np.ones((2,3)))
    assert np.array_equal(a.grad, np.ones((2,3)))
    assert np.array_equal(b.grad, np.array([2,2,2]))

def test_broadcast_add_backward2():
    a = loupe.rand(size=(2,3), requires_grad=True)
    b = loupe.rand(size=(1,3), requires_grad=True)
    res = a + b 
    res.backward(grad=np.ones((2,3)))
    assert np.array_equal(a.grad, np.ones((2,3)))
    assert np.array_equal(b.grad, np.array([[2,2,2]]))
