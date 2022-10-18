import pytest
import numpy as np
import loupe


@pytest.mark.parametrize("data", ['test', len])
def test_asarray_invalid_dtype(data):
    with pytest.raises(TypeError):
        loupe.asarray(data)


def test_array_creation_utils():
    assert loupe.ones((5, 5))
    assert loupe.zeros((5, 5))
    assert loupe.rand(size=(5, 5))
    assert loupe.randn(size=(5, 5))


def test_array_like_utils():
    x = loupe.ones((5, 5))
    y = loupe.ones_like(x)
    z = loupe.zeros_like(x)

    assert y.shape == x.shape
    assert z.shape == x.shape


def test_shift():
    a = np.zeros((3, 3))
    a[2, 2] = 1
    s = loupe.shift(a, (-1, -1))
    assert np.allclose(s, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))


def test_register():
    ref = np.zeros((3, 3))
    ref[1, 1] = 1
    arr = np.zeros((3, 3))
    arr[2, 2] = 1
    s, e = loupe.register(arr, ref, oversample=10, return_error=True)
    assert np.array_equal(s, [-1, -1])
    assert np.allclose(e, 0, atol=1e-6)


def test_medfix2():
    a = np.random.normal(size=(3,3))
    mask = np.zeros_like(a)
    mask[1,1] = 1
    mask = np.asarray(mask, dtype=bool)
    assert(np.array_equal(loupe.medfix2(a, mask)[1,1], np.median(a[~mask])))
