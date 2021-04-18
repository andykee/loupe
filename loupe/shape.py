import numpy as np


def mesh(shape, shift=(0, 0)):
    """Generate a standard mesh."""

    nr = shape[0]
    nc = shape[1]

    x = np.arange(shape[1]) - np.floor(shape[1]/2.0) - shift[1]
    y = np.arange(shape[0]) - np.floor(shape[0]/2.0) - shift[0]

    return np.meshgrid(x, y)


def circle(shape, radius, shift=(0, 0)):
    """Compute a circle with anti-aliasing.

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)

    radius : float
        Radius of circle in pixels

    shift : (2,) array_like, optional
        How far to shift center in float (rows, cols). Default is (0, 0).

    Returns
    -------
    out : ndarray

    """
    x, y = mesh(shape)
    r = np.sqrt(np.square(x - shift[1]) + np.square(y - shift[0]))
    return np.clip(radius + 0.5 - r, 0.0, 1.0)


def circlemask(shape, radius, shift=(0, 0)):
    """Compute a circular mask.

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)

    radius : float
        Radius of circle in pixels

    shift : array_like, optional
        How far to shift center in float (rows, cols). Default is (0, 0).

    Returns
    -------
    out : ndarray

    """

    mask = circle(shape, radius, shift)
    mask[mask > 0] = 1
    return mask


def hexagon(shape, radius, rotate=False):
    """Compute a hexagon mask.

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)

    radius : int
        Radius of outscribing circle (which also equals the side length) in
        pixels.

    rotate : bool
        Rotate mask so that flat sides are aligned with the Y direction instead
        of the default orientation which is aligned with the X direction.

    Returns
    -------
    out : ndarray

    """

    inner_radius = radius * np.sqrt(3)/2
    side_length = radius/2

    y, x = mesh(shape)

    rect = np.where((np.abs(x) <= side_length) & (np.abs(y) <= inner_radius))
    left_tri = np.where((x <= -side_length) & (x >= -radius) & (np.abs(y) <= (x + radius)*np.sqrt(3)))
    right_tri = np.where((x >= side_length) & (x <= radius) & (np.abs(y) <= (radius - x)*np.sqrt(3)))

    mask = np.zeros(shape)
    mask[rect] = 1
    mask[left_tri] = 1
    mask[right_tri] = 1

    if rotate:
        return mask.transpose()
    else:
        return mask


def slit(shape, width):
    """Compute a horizontal slit mask.

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)

    width : int
        Slit width in pixels

    Returns
    -------
    out : ndarray

    """

    y, x = mesh(shape)

    mask = np.zeros(shape)
    mask[np.abs(x) <= width/2] = 1

    return mask