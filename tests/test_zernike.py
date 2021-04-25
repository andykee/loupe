import pytest
import numpy as np
import loupe


def test_zernike_rho_theta():
    with pytest.raises(ValueError):
        loupe.zernike(mask=1, index=1, normalize=True, rho=1, theta=None)


def test_zernike_basis():
    basis = loupe.zernike_basis(mask=np.ones((3, 3)), modes=1, vectorize=False)
    assert np.array_equal(basis, np.ones((1, 3, 3)))


def test_zernike_basis_vectorize():
    basis = loupe.zernike_basis(mask=np.ones((3, 3)), modes=1, vectorize=True)
    assert np.array_equal(basis, np.ones((1, 9)))


def test_zernike_fit():
    mask = loupe.circlemask((256, 256), 128)
    coeffs = np.random.rand(4)*100e-9
    phase = loupe.zernike_compose(mask, coeffs)
    fit_coeffs = loupe.zernike_fit(phase, mask, np.arange(1, 5))
    assert np.all(np.isclose(coeffs, fit_coeffs))


def test_zernike_remove():
    mask = loupe.circlemask((256, 256), 128)
    coeffs = np.random.rand(4)*100e-9
    phase = loupe.zernike_compose(mask, coeffs)
    residual = loupe.zernike_remove(phase, mask, np.arange(1, 5))
    assert np.all(np.isclose(residual, np.zeros_like(residual)))
