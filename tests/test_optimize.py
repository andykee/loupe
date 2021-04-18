import numpy as np
import loupe


def test_optimize_scalar_quadratic():
    x = loupe.rand()
    y = 4*(x-5)**2
    res = loupe.optimize(y, x)
    assert np.allclose(res.x, 5)


def test_optimize_scalar_quadratic_finite_difference():
    x = loupe.rand()
    y = 4*(x-5)**2
    res = loupe.optimize(y, x, analytic_grad=False)
    assert np.allclose(res.x, 5)
