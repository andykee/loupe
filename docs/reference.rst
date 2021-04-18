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

Optimization helpers
--------------------
.. autosummary::
    :toctree: generated/

    gradcheck
    finite_difference_grad

Function API
------------
.. autosummary::
    :toctree: generated/
    :template: Function.rst
    
    core.Function