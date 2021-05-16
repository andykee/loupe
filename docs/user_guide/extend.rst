.. _extend:

***************
Extending Loupe
***************

Adding a new operation to Loupe requires implementing a new 
:class:`~loupe.core.Function` class. Any new `Function` should do three 
things:

1. Populate the computational graph with details of the operation
2. Implement a :func:`~loupe.core.Function.forward` method to compute the 
   results of the operation
3. Implement a :func:`~loupe.core.Function.backward` method to compute 
   gradients as needed during a backward pass


Populating the computational graph
==================================
New functions should inherit from :class:`loupe.core.Function`. The class
constructor can take as many arguments as needed to perform the operation, and
may provide default values. Because access to the inputs is needed by other 
class methods, they should be stored as instance attributes by the class 
constructor. Inputs are registered to the computational graph by passing them 
to the :class:`~loupe.core.Function` class constructor.

Any Python type may be accepted, but only :class:`loupe.array` or 
:class:`~loupe.core.Function` objects may be tracked on the computational 
graph. The :func:`~loupe.asarray` method can be used to ensure inputs 
implement the `array` interface.

Note that only inputs that may be optimization parameters need to be 
registered on the graph. There is no harm in registering all inputs, but doing
so may impact overall performance.

As a motivating example, we'll recreate the :class:`~loupe.power` function 
here step-by-step while providing additional comments to describe the 
underlying logic:

.. code:: python

    class power(loupe.core.Function):
        def __init__(self, x1, x2):
            # Store input as instance attribute. Because the inpput may vary
            # during optimization, it must be array-like
            self.input = loupe.asarray(x1)

            # Because the exponent will not change once it is defined, it does
            # not need to be array-like and can be stored in its native format
            self.exp = x2

            # Register input on the graph. Since the exponent will not
            # vary during optimization, there is no need to register it
            # on the graph.
            super().__init__(self.input)


``forward()``
=============
:func:`~loupe.core.Function.forward` is called any time the function needs to
be evaluated. The result of the operation is computed using the inputs passed 
to the `Function` constructor and the computed result should be returned as a 
Numpy array.

Because some of the function inputs may be functions themselves, it is 
necessary to explicitly grab the underlying data from any Loupe arrays or
functions using either the :attr:`~loupe.array.data` attribute or the 
:func:`~loupe.array.getdata()` method. Because this operation may be 
computationally expensive, it should only be performed once during function
evaluation. 

Some of the input data used in `forward()` may also be needed in `backward()`. 
Since fetching the input data can be expensive, it is desirable to cache the
results so they do not need to be recomputed when `backward()` is called. The
:func:`~loupe.core.Function.cache_for_backward` method provides a simple 
list-based caching mechanism for this purpose.

Returning to our implementation of `power()`, we'll write the code to compute
:math:`y = x^n`:

.. code:: python

    class power(loupe.core.Function):
        def __init__(self, x1, x2):
            self.input = loupe.asarray(x1)
            self.exp = x2
            super().__init__(self.input)

        def forward(self):
            # Get the input's underlying data
            input = self.input.getdata()

            # Cache input's data for use in backward()
            self.cache_for_backward(input)

            # Compute and return the result. Note we can use self.exp
            # directly since it constant once defined in __init__()
            return np.power(input, self.exp)

Note that both ``data`` and ``getdata()`` (when called with the default 
arguments) return the same underlying data, but ``getdata()`` provides 
additional flexibility if needed and is therefore the preferred approach.

``backward()``
==============
:func:`~loupe.core.Function.backward` is called any time the gradient of the
function needs to be computed. The ``grad`` argument is always provided and 
represents the gradient with respect to the given output of the function.
`backward()` should compute the gradient of each function input with 
``requires_grad=True`` and pass the result to the input's `backward()` method.

Note that any inputs that were cached in `forward()` are available in the 
function's :attr:`~loupe.core.Function.cache` attribute. Because `cache` is a 
Python list, list unpacking is the recommended approach for accessing the 
cached values.

We'll complete our development of the :class:`~loupe.power` function by 
implementing its `backward()` method to compute 
:math:`\bar{x} = n x^{n-1} \bar{y}`:

.. code:: python

    class power(loupe.core.Function):
        def __init__(self, x1, x2):
            self.input = loupe.asarray(x1)
            self.exp = x2
            super().__init__(self.input)

        def forward(self):
            input = self.input.getdata()
            self.cache_for_backward(input)
            return np.power(input, self.exp)

        def backward(self, grad):
            # Unpack the value of input that was cached during the
            # call to forward()
            input, = self.cache

            # Compute the gradient of this function and pass the
            # result to input.backward()
            grad = self.exp * np.power(input, self.exp-1) * grad
            self.input.backward(grad)
