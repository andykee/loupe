import numpy as np
import pytest

import loupe

@pytest.mark.parametrize('a, b, axes', [
    (
        np.random.uniform(size=(3, 5, 5)),
        np.random.uniform(size=3),
        (0,0)
    )
])
def test_tensordot(a, b, axes):
    assert np.array_equal(loupe.tensordot(a, b, axes), np.tensordot(a, b, axes))
