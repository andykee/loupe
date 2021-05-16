import numpy as np
import loupe

np.seterr(all='raise')

s = 'abcdefghijklmnopqrstuvwxyz'


class sserror(loupe.core.Function):
    r"""Compute the normalized sum squared error between two arrays.

    The normalized sum squared error between ideal data `f` and estimated data
    `g` is given by

    .. math::

        \mbox{error} = \frac{1}{N}\sum_n \frac{\sum{\left|g(n,x,y)-f(n,x,y)\right|^2}}{\sum{\left|f(n,x,y)\right|^2}}

    for :math:`n = 1, 2, ... N` independent measurements.

    If `gain_bias_invariant` is ``True``, the error metric is computed such 
    that the result is independent of relative scaling (gain) and offset 
    (bias) differences between `f` and  `g` [1].
    
    Parameters
    ----------
    model : array_like
        Estimated or modeled data
    data : array_like
        Ideal or measured data
    mask : array_like, optional
        Mask applied to the inputs where a True value indicates that the 
        corresponding element of the array is invalid. Mask must either have
        the same shape as the inputs or be broadcastable to the same shape and
        contain entries that are castable to bool. If None (default), the 
        inputs are not masked.
    gain_bias_invariant : bool, optional
        If True (default), the error is computed to be independent of any 
        gain or bias differences between the inputs. 

    Returns
    -------
    error : float
        Normalized root mean squared error

    References
    ----------
    [1] S. Thurman and J. Fienup, "Phase retrieval with signal bias", J. Opt. Soc. Am. A/Vol. 26, No. 4 (2009)
    
    [2] A. Jurling and J. Fienup, "Applications of algorithmic differentiation to phase retrieval algorithms", J. Opt. Soc. Am. A/Vol. 31, No. 7 (2014)
    
    """
    def __init__(self, model, data, mask=None, gain_bias_invariant=True):
        self.f = np.asarray(data)
        self.g = loupe.asarray(model)

        if mask is not None:
            if mask.shape == data.shape:
                self.mask = mask
            else:
                self.mask = np.broadcast_to(mask, self.f.shape)
        else:
            self.mask = mask

        self.gain_bias_invariant = gain_bias_invariant

        # _f, _K, and _D only depend on static input data f and the mask, so
        # we can precompute them here
        self._f = self._G(self.f, self.mask)
        
        if self.f.ndim == 3:
            self._K = self.f.shape[0]
        else:
            self._K = 1
        
        if self.mask is None:
            self._D = np.sum(self.f ** 2)
        else:
            self._D = np.sum(self.f ** 2 * self.mask)

        super().__init__(self.g)
        
    @staticmethod
    def _G(data, mask):
        # This function implements Eqs. 12 and 13 in [1]
        if mask is None:
            numer = np.sum(data, axis=(-2, -1))
            denom = np.prod(data.shape[-2:])
        else:
            numer = np.sum(mask * data, axis=(-2, -1))
            denom = np.sum(mask, axis=(-2, -1))
        x = numer/denom
        if data.ndim == 3:
            x = x[:, np.newaxis, np.newaxis]
        return data - x

    @staticmethod
    def _fit_gain(_f, _g, mask):
        # This function implements Eq. 14 in [1]
        try:
            if mask is None:
                numer = _f * _g
                denom = np.square(_g)
            else:
                numer = mask * _f * _g
                denom = mask * np.square(_g)
            
            return np.sum(numer, axis=(-2,-1))/np.sum(denom, axis=(-2,-1))

        except FloatingPointError as e:
            if np.all(_g == 0):
                raise ZeroDivisionError('g must be nonzero')
            else:
                raise e

    @staticmethod
    def _fit_bias(f, g, mask, gain):
        # This function implements Eq. 10 in [1]
        gain = np.asarray(gain)
        subscripts = f'{s[0:g.ndim]},{s[0:gain.ndim]}->{s[0:g.ndim]}'
        resid = f - np.einsum(subscripts, g, gain)
        if mask is None:
            numer = np.sum(resid, axis=(-2, -1))
            denom = np.prod(f.shape[-2:])
        else:
            numer = np.sum(mask*resid, axis=(-2, -2))
            denom = np.sum(mask, axis=(-2, -1))

        return numer/denom

    def residual(self):
        """Compute the residual error between `f` and `g`.

        Returns
        -------
        resid : ndarray
            Residual error between `f` and `g`.
        """
        g = self.g.getdata()

        if self.gain_bias_invariant:
            _g = self._G(g, self.mask)
            alpha = self._fit_gain(self._f, _g, self.mask)

            subscripts = f'{s[0:_g.ndim]},{s[0:alpha.ndim]}->{s[0:_g.ndim]}'

            if self.mask is None:
                resid = np.einsum(subscripts, _g, alpha) - self._f 
            else:
                resid = self.mask * (np.einsum(subscripts, _g, alpha) - self._f)
        else:
            if self.mask is None:
                resid = g - self.f
            else:
                resid = self.mask * (g - self.f)

        return resid

    def forward(self):
        g = self.g.getdata()

        if self.gain_bias_invariant:
            _g = self._G(g, self.mask)
            alpha = self._fit_gain(self._f, _g, self.mask)

            subscripts = f'{s[0:_g.ndim]},{s[0:alpha.ndim]}->{s[0:_g.ndim]}'

            if self.mask is None:
                resid = np.einsum(subscripts, _g, alpha) - self._f
                sse = 1/self._K * np.sum(np.square(resid))/np.sum(np.square(self.f))
            else:
                resid = self.mask * (np.einsum(subscripts, _g, alpha) - self._f)
                sse = 1/self._K * np.sum(np.square(resid))/np.sum(self.mask*np.square(self.f))

        else:
            alpha = np.array(1.0)
            if self.mask is None:
                resid = g - self.f
                numer = np.sum(np.square(resid), axis=(-2, -1))
                denom = np.sum(np.square(self.f), axis=(-2, -1))
                sse = 1/self._K * np.sum(numer/denom)
            else:
                resid = self.mask * (g - self.f)
                numer = np.sum(np.square(resid), axis=(-2, -1))
                denom = np.sum(self.mask * np.square(self.f), axis=(-2, -1))
                sse = 1/self._K * np.sum(numer/denom)

        self.cache_for_backward(resid, alpha)

        return float(sse)

    def backward(self, grad):
        # This function implements Eq. 53 in [2]
        resid, alpha = self.cache
        subs = f'{s[0:resid.ndim]},{s[0:alpha.ndim]}->{s[0:resid.ndim]}'

        if self.mask is None:
            grad = np.einsum(subs, resid, (2/self._D) * grad * alpha)
        else:
            grad = np.einsum(subs, self.mask * resid, (2/self._D) * grad * alpha)

        self.g.backward(grad)
