.. _api:

*************
API Reference
*************

.. currentmodule:: loupe

Arrays
======

.. autosummary::
    :toctree: generated/

    array

Array creation
--------------
.. autosummary::
    :toctree: generated/

    asarray
    ones
    ones_like
    zeros
    zeros_like
    rand
    randn
    
Universal functions
===================

Math operations
---------------
.. autosummary::
    :toctree: generated
    :template: ufunc.rst

    add
    subtract
    multiply
    power
    einsum

Cost functions
--------------
.. autosummary::
    :toctree: generated/
    :template: ufunc.rst

    sserror

Optimization
============

Optimization functions
----------------------
.. autosummary::
    :toctree: generated/

    optimize

Function API
------------
.. autosummary::
    :toctree: generated/
    :template: Function.rst
    
    core.Function

Utilities
=========

Shapes
------
.. autosummary::
    :toctree: generated/

    circle
    circlemask
    hexagon
    slit

Zernike polynomials
-------------------
.. autosummary::
    :toctree: generated/

    zernike
    zernike_basis
    zernike_compose
    zernike_fit
    zernike_remove