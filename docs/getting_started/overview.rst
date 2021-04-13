.. _overview:

********
Overview
********

Loupe is a Python library for defining and solving optimization problems by 
minimizing scalar cost functions. It was originally developed to solve 
parametric `phase retrieval <https://en.wikipedia.org/wiki/Phase_retrieval>`_ 
problems but is able to solvea wide variety of general optimization problems 
as well.

Loupe was designed to be:

* *User friendly* - provide a simple, consistent interface for common use cases.
  Loupe is generally compatible with Numpy arrays and where possible, Loupe mimics 
  Numpy's API so if you know how to use Numpy, you know how to use Loupe. 
* *Fast* - forward models are defined and manipulated dynamically. These operations
  are automatically tracked in a computational graph, making it possible to 
  analytically compute gradients which can then be fed to an optimizer - greatly
  speeding up optimization convergence.
* *Easy to extend* - the core function object is designed to be 
  subclassed, making it very easy to develop and use new operations

Importing Loupe
===============
Every public function and class is available in Loupe's root namespace. Loupe's 
standard library can be imported as follows:

.. code:: pycon

    >>> import loupe

Getting help
============
The best place to ask for help on subjects not covered in this documentation or suggest new 
features/ideas is by opening a ticket on `Github <https://github.com/andykee/loupe/issues>`__