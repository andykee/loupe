import numpy as np
import loupe

def test_exp():
    a = loupe.rand(size=(10,10))
    res = loupe.exp(a)
    assert np.array_equal(res, np.exp(a.data))

def test_exp_backward():
    a = loupe.rand(size=(10,10), requires_grad=True)
    res = loupe.exp(a)
    res.backward(grad=np.ones((10,10)))
    assert np.array_equal(a.grad, res) 
