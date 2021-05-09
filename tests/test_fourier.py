import numpy as np
import pytest

import loupe


@pytest.mark.parametrize("n", [10, 11])
def test_dft2(n):

    f = np.random.rand(n, n) + 1j * np.random.rand(n, n)

    F_dft = loupe.dft2(f, (1/n, 1/n), unitary=False)
    F_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f)))

    assert np.allclose(F_dft, F_fft)


@pytest.mark.parametrize("n", [10, 11])
def test_dft2_backward(n):

    f = loupe.rand(size=(n, n), requires_grad=True)
    F = loupe.dft2(f, (1/n, 1/n), unitary=False)

    grad = np.ones(shape=(n, n))
    F.backward(grad)

    g = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grad))) 

    assert np.allclose(f.grad, g.real * n**2)


def test_dft2_unitary():
    n = 10
    f = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    F = loupe.dft2(f, (1/n, 1/n), unitary=True)

    assert np.allclose(np.sum(np.abs(f)**2), np.sum(np.abs(F)**2))