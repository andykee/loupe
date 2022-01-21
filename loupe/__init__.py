__version__ = '1.1.0'

from loupe.core import array

from loupe.cost import sserror

from loupe.einsum import einsum

from loupe.fourier import dft2

from loupe.numeric import (
    add,
    subtract,
    multiply,
    power,
    exp,
    expc,
    sum,
    slice
)

from loupe.optimize import optimize

from loupe.shape import circle, circlemask, hexagon, slit

from loupe.special import absolute_square, abs_square

from loupe.tensordot import tensordot

from loupe.util import (
    asarray,
    zeros,
    zeros_like,
    ones,
    ones_like,
    rand,
    randn,
    centroid,
    shift,
    register
)

from loupe.zernike import (
    zernike,
    zernike_compose,
    zernike_basis,
    zernike_fit,
    zernike_remove,
    zernike_coordinates
)
