from collections import namedtuple

import numpy as np
import scipy.optimize

import loupe

def optimize(fun, params, ftol=1e-9, gtol=1e-5, maxiter=1000,
             analytic_grad=True):
    """Minimize a scalar function of one or more variables using the
    L-BFGS-B algorithm.

    This function constructs a call to `scipy.optimize.minimize() 
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
    with `method = 'L-BFGS-B'`.

    Parameters
    ----------
    fun : :class:`~loupe.core.Function`
        The objective function to be minimized.
    params : list of :class:`array` or :class:`array`
        Parameters in `fun` to optimize.
    ftol : float, optional
        Termination tolerance on the objective function value. Default is 1e-9.
    gtol : float, optional
        Termination tolerance on the gradient value. Default is 1e-6.
    maxiter : int, optional
        Maximum number of iterations. Default is 1000.
    analytic_grad : bool, optional
        If true (default), the gradient vector is analytically computed using
        algorithmic differentiation. If false, the gradient is numerically
        estimated by a finite difference algorithm.

    Returns
    -------
    res : OptimizeResult
        The optimization result.  Important attributes are: ``x`` the solution 
        array, ``success`` a Boolean flag indicating if the optimizer exited 
        successfully and ``message`` which describes the cause of the 
        termination. See `OptimizeResult 
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html>`_ 
        for a full description of other attributes.

    Examples
    --------
    Construct a simple quadratic objective function and compute the minimum
    value of x:

    .. code:: pycon

        >>> x = loupe.rand()
        >>> y = 4*(x-5)**2
        >>> loupe.optimize(y, params=x)

             fun: array(1.26217745e-29)
        hess_inv: <1x1 LbfgsInvHessProduct with dtype=float64>
             jac: array([-1.42108547e-14])
         message: 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
            nfev: 3
             nit: 2
            njev: 3
          status: 0
         success: True
               x: array([5.])

    Optimize the same function, this time using the finite difference 
    gradient:

    .. code:: pycon

        >>> x = loupe.rand()
        >>> y = 4*(x-5)**2
        >>> loupe.optimize(y, params=x, analytic_grad=False)

             fun: array(8.4112054e-13)
        hess_inv: <1x1 LbfgsInvHessProduct with dtype=float64>
             jac: array([3.70850496e-06])
         message: 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
            nfev: 6
             nit: 2
            njev: 3
          status: 0
         success: True
               x: array([5.00000046])
               
    """
    optim = Blueprint(fun, params)

    options = {'ftol': ftol, 'gtol': gtol, 'maxiter': maxiter}

    if analytic_grad:
        jac = optim.grad
    else: 
        jac = None

    res = scipy.optimize.minimize(optim.cost, optim.x, method='L-BFGS-B', 
                                  jac=jac, options=options)

    # NOTE: the solution returned by scipy.optimize.minimize is always one
    # step prior to the final step. (is this true for optimizers other than
    # L-BGFS-B or when the number of iterations constraint is hit?)

    # this means that optim.x is sometimes set in a much worse way than the
    # minimum solution. We'll upadte x here with the result
    optim.x = res.x

    return res


class Blueprint:
    def __init__(self, fun, params):

        params = [params] if not isinstance(params, (list, tuple)) else params
        _mark_params_for_optimization(fun, params)

        self.fun = fun
        self.params = []
        index = 0
        for p in params:
            size = p.flatten(apply_mask=True).size
            self.params.append((p, slice(index, index+size)))
            index += size

    @property
    def x(self):
        x = np.empty(self.params[-1][1].stop, dtype=np.float64)
        for param in self.params:
            x[param[1]] = param[0].flatten(apply_mask=True)
        return x

    @x.setter
    def x(self, value):
        if value is not None:
            value = np.asarray(value)
            for param in self.params:
                if param[0].mask is not None:
                    # Optimizer only knows about valid values here. Need to
                    # dump these in the right spot in the full array
                    param[0][~param[0].mask] = value[param[1]]
                else:
                    param[0][...] = value[param[1]].reshape(param[0].shape) 


    def cost(self, x=None):
        # Evaluate the cost function at x
        if x is not None:
            if not np.all(x == self.x):
                self.x = x
        return self.fun.getdata()
    
    def grad(self, x=None):
        # Compute the gradient of all optimization parameters at x and 
        # return the result as a flattened array suitable for
        # consumption by an optimizer
        if x is not None: 
            if not np.all(x == self.x):
                self.x = x
        self.zero_grad()
        self.fun.backward(np.array(1.))
        
        grad = np.zeros(self.params[-1][1].stop, dtype=np.float64)
        for param in self.params:
            grad[param[1]] = param[0].grad_flatten(apply_mask=True)
        return grad

    def zero_grad(self):
        # Zero the gradients of all arrays in the graph
        for arr in _dump_arrays(self.fun):
            arr.zero_grad()


def _mark_params_for_optimization(fun, params):
    # Recursively trace through the graph and mark parameters to optimize
    # with requires_grad = True. All others should have 
    # requires_grad=False
    for arr in _dump_arrays(fun):
        if arr in params:
            arr.requires_grad = True
        else:
            arr.requires_grad = False  


def _dump_arrays(node, out=None):

    if out is None:
        out = []
    if isinstance(node, loupe.core.Function):
        for n in node.inputs:
            _dump_arrays(n, out)
    else:
        if node not in out:
            out.append(node)
    return out
