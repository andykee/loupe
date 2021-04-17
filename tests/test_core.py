import numpy as np
import loupe


def test_backward_requires_grad_false():
    a = loupe.array([1,2,3], requires_grad=False)
    a.backward([1,1,1])
    assert np.all(a.grad == [0,0,0])