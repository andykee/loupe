import numpy as np
import loupe

def test_power():
    a = loupe.rand(size=(10,10))
    b = loupe.rand(size=(10,10))
    res = loupe.power(a,b)
    assert np.array_equal(res, np.power(a.data, b.data))

def test_power_overloaded():
    a = loupe.rand(size=(10,10))
    b = loupe.rand(size=(10,10))
    res = a ** b
    assert np.array_equal(res, np.power(a.data, b.data))

def test_scalar_power():
    a = loupe.rand(size=(10,10))
    b = loupe.rand()
    res = a ** b
    assert np.array_equal(res, np.power(a.data, b.data)) 

def test_power_backward():
    a = loupe.rand(size=(10,10), requires_grad=True)
    b = np.random.uniform(size=(10,10))
    res = loupe.power(a,b)
    res.backward(grad=np.ones((10,10)))
    assert np.array_equal(a.grad, b * np.power(a.data, b-1)) 

def test_scalar_power_backward():
    a = loupe.rand(size=(10,10), requires_grad=True)
    b = 3
    res = loupe.power(a,b)
    res.backward(grad=np.ones((10,10)))
    assert np.array_equal(a.grad, b * np.power(a.data, b-1)) 

