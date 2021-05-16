.. _user_guide.functions:

*********
Functions
*********

Introduction
============
Every operation performed on an ``array`` object creates a new ``Function``
object. The returned ``Function`` object implements the same interface as 
``array``, but also adds a record of the operation to the 
:ref:`computational graph <autograd>`.


Math operations
===============
Loupe defines functions for many common math operations. These functions are
generally applied *elementwise* to an array. For example:

.. code:: pycon

    >>> import loupe
    >>> a = loupe.array([1,2,3])
    >>> b = loupe.array([4,5,6])
    >>> c = loupe.add(a, b)
    >>> c
    array([5., 7., 9.])

Common infix notation for ``+``, ``-``, ``*``, and ``**`` is also supported:

.. code:: pycon

    >>> d = a + b
    >>> d
    array([5., 7., 9.])

The available math operations are provided below:

.. currentmodule:: loupe
.. autosummary::

    add
    subtract
    multiply
    power
    exp
    expc
    sum
    absolute_square
    dft2
    einsum
    tensordot    


Array manipulation
==================

.. currentmodule:: loupe
.. autosummary::

    slice

.. Special functions
.. =================


Broadcasting
============
Broadcasting allows functions to  operate on arrays that do not have exactly
the same shape. Subject to certain constraints, the smaller array is 
“broadcast” across the larger array so that the resulting arrays have 
compatible shapes. This operation is done without making needless copies of
the underlying data. 

See Numpy's page on `broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_
for more details, examples, and a description of the broadcasting rules.

