import numpy as np

# This function was implemented in Numpy v1.12 and hasn't
# substantively changed since then (Numpy v1.20)
from numpy.core.einsumfunc import _parse_einsum_input

import loupe


# The implementation here is heavily influenced by chainer's einsum(). It
# has been simplified a bit and adapted to work with loupe's interfaces.

class einsum(loupe.core.Function):
    """
    einsum(subscripts, *operands, dtype=None)

    Evaluates the Einstein summation convention on the operands.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as comma separated list of subscript
        labels. An implicit (classical Einstein summation) calculation is performed 
        unless the explicit indicator ‘->’ is included as well as subscript labels 
        of the precise output form.
    operands : list of array_like
        The arrays used to perform the requested operation.

    Returns
    -------
    out : :class:`~loupe.core.Function`
        The result of the requested Einstein summation operation.
    
    See also
    --------
    :class:`~loupe.tensordot`

    """
    def __init__(self, *operands, dtype=None):
        # NOTE: I'm not sure this will work with complex inputs
        
        # Use numpy's _parse_einsum_input to tear apart the subscripts. Note
        # that although _parse_einsum_input also returns the operands, it 
        # casts them to numpy arrays which is not desirable here so we'll
        # just manually yank out the operands to preserve any loupe objects.
        in_subs, out_sub, _ = _parse_einsum_input(operands)
        opers = operands[1:] 

        self.in_subs = in_subs
        self.out_sub = out_sub
        self.operands = [loupe.asarray(oper) for oper in opers]
        
        super().__init__(*self.operands)
        
    @property
    def _inputs_requires_grad(self):
        # Return the indices of inputs with requires_grad = True
        return [n for n, s in enumerate(self.inputs) if s.requires_grad]
    
    def forward(self):
        subscripts = f'{self.in_subs}->{self.out_sub}'
        operands = [oper.getdata() for oper in self.operands]
        result = np.einsum(subscripts, *operands)
        
        self.cache_for_backward(*operands, result)
   
        return result
        
    def backward(self, grad):
        in_subs_fw = self.in_subs.split(',')
        out_sub_fw = self.out_sub
        
        *inputs, result = self.cache
        indices = self._inputs_requires_grad
        
        # compute gradient for inputs with requires_grad=True
        for i in indices:
            in_subs = ','.join([(out_sub_fw if j == i else s) for j, s in enumerate(in_subs_fw)])
            out_sub = in_subs_fw[i]
            out_shape = inputs[i].shape
            
            diag_inputs = [(grad if j == i else x) for j, x in enumerate(inputs)]
            out_set = set(out_sub)
            io_set = out_set.intersection(set(in_subs))
            
            if len(io_set) == len(out_sub):
                subscripts = f'{in_subs}->{out_sub}'
                y = np.einsum(subscripts, *diag_inputs)
            else:
                # I'm not sure what kind of expression will get us here ¯\_ (ツ)_/¯ 
                # See chainer DiagEinsum else block for implementation if it's needed
                raise NotImplementedError
                
            self.inputs[i].backward(y)

