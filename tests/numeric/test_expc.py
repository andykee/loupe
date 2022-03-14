import numpy as np
import loupe


def test_expc():
    a = loupe.rand(size=(10, 10))
    res = loupe.expc(a)
    assert np.allclose(res, np.exp(a.data*1j))


def test_expc_backward():
    a = loupe.rand(size=(10, 10), requires_grad=True)
    res = loupe.expc(a)
    res.backward(grad=np.ones((10, 10)))
    assert np.allclose(a.grad, np.imag(np.conj(res)))
