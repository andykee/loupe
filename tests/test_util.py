import pytest
import loupe


@pytest.mark.parametrize("data",['test', len])
def test_asarray_invalid_dtype(data):
    with pytest.raises(TypeError):
        loupe.asarray(data)

def test_array_creation_utils():
    assert loupe.ones((5,5))
    assert loupe.zeros((5,5))
    assert loupe.rand(size=(5,5))
    assert loupe.randn(size=(5,5))

def test_array_like_utils():
    x = loupe.ones((5,5))
    y = loupe.ones_like(x)
    z = loupe.zeros_like(x)

    assert y.shape == x.shape
    assert z.shape == x.shape
