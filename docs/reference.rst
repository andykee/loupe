.. _api:

.. currentmodule:: loupe

*************
API Reference
*************

.. _api.array:

Array
=====

.. autosummary::
    :toctree: generated/
    :caption: Array

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
    

.. _api.functions:

Functions
=========

Math operations
---------------
.. autosummary::
    :toctree: generated
    :template: function.rst
    :caption: Functions

    add
    subtract
    multiply
    power
    exp
    expc
    absolute_square
    einsum
    tensordot

Array manipulation
------------------
.. autosummary::
    :toctree: generated/
    :template: function.rst

    slice

Cost functions
--------------
.. autosummary::
    :toctree: generated/
    :template: function.rst

    sserror

Function API
------------
.. autosummary::
    :toctree: generated/
    
    core.Function

Optimization
============

.. autosummary::
    :toctree: generated/
    :caption: Optimization

    optimize

Utilities
=========

Shapes
------
.. autosummary::
    :toctree: generated/
    :caption: Utilities

    circle
    circlemask
    hexagon
    slit

Image tools
-----------
.. autosummary::
    :toctree: generated/

    centroid
    shift
    register

Zernike polynomials
-------------------
.. autosummary::
    :toctree: generated/

    zernike
    zernike_basis
    zernike_compose
    zernike_fit
    zernike_remove