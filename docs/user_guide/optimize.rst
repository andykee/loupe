.. _optimize:

************
Optimization
************

Loupe provides a suite of tools for building models and optimizing their
parameters. Optimization is performed using the L-BFGS-B algorithm, which 
attempts to find values :math:`\mathbf{x}` that minimize :math:`f(\mathbf{x})`
where :math:`f` is a differentiable scalar cost function.

This section of the user guide describes how to construct a model, define a
cost function, and use the optimizer to find a set of parameters that minimize
the cost function. 

Creating models
===============
Loupe models are constructed using a "define-by-run" approach. This means that 
a model is created by simply writing the code that defines the actual forward 
computation using Loupe :ref:`array <user_guide.array>` and :ref:`Function 
<user_guide.functions>` objects. Because models are created on the fly, they 
can be easily validated, debugged, and updated using simple numerical 
programming. 

For example, suppose we have some process that can be represented by 
:math:`y = Ax^2 + x` where `A` is a known 3 x 3 matrix and `x` is a vector 
with size 3. For this example, we'll randomly compute the `A` matrix.

.. code:: pycon

    >>> import numpy as np
    >>> import loupe
    >>> A = np.random.uniform(low=-1, high=1, size=(3,3))
    >>> x = loupe.zeros(3)
    >>> y = A * x ** 2 + x

Note that we've defined ``x`` with a set of initial values (zeros in this 
case) that will serve as the starting guess for the optimizer.

Defining cost functions
=======================
A cost function (sometimes also called an objective function) is a 
mathematical expression that represents some quantity to be minimized. In most
cases, the cost function computes the error between the model results and some
measured data or desired result. 

.. note::

    Loupe's optimizer requires that all cost functions return a scalar value.

For the purposes of this example, we'll establish the true value of ``x`` and
generate some synthetic data:

.. code:: pycon

    >>> x_true = np.array([1, -2, 3])
    >>> data = A * x_true ** 2 + x_true

Now we'll construct a cost function. A commonly used metric is the sum squared
error, given by :math:`\mbox{err} = \sum (x - f(x))^2`. This is implemented in 
code below:

.. code:: pycon

    >>> err = loupe.sum((data - y)**2)

If a more full-featured cost function is required, see :class:`~loupe.sserror`.

Optimizing model parameters
===========================
Optimizing the model parameters is as simple as passing a cost function and
the model parameter or parameters to be optimized to the 
:func:`~loupe.optimize` function. 

Completing our example from above:

.. code:: pycon

    >>> loupe.optimize(fun=cost, param=x)
         fun: array(1.00560419e-13)
    hess_inv: <3x3 LbfgsInvHessProduct with dtype=float64>
         jac: array([-9.61103295e-07,  1.48569373e-06,  2.01840447e-06])
     message: b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
        nfev: 22
         nit: 12
        njev: 22
      status: 0
     success: True
           x: array([ 0.99999986, -1.99999997,  3.00000001])

We see that the values of `x` estimated by the optimizer agree very closely
with the true values established earlier.

.. note::

    The default behaavior of `optimize` is to compute the analytic gradient
    at each iteration and provide it to the underlying optimization 
    algorithm. If this is not desired for some reason, calling `optimize` with
    ``analytic_grad=False`` will estimate the gradient using a finite 
    difference algorithm instead. Note that this will impact convergence 
    speed.