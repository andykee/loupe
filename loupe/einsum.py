import numpy as np

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
    out : Function
        The result of the requested Einstein summation operation.
        

    """
    def __init__(self, *operands, dtype=None):
        # NOTE: I'm not sure this will work with complex inputs
        
        in_subs, out_sub, opers = _parse_einsum_input(operands)
        
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
        operands = [oper.data for oper in self.operands]
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


einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
einsum_symbols_set = set(einsum_symbols)


# This is a very slightly modified version of an identically named function in Numpy v1.19.0
def _parse_einsum_input(operands):
    """
    A reproduction of einsum c side einsum parsing in python.
    Returns
    -------
    input_strings : str
        Parsed input strings
    output_string : str
        Parsed output string
    operands : list of array_like
        The operands to use in the numpy contraction
    Examples
    --------
    The operand list is simplified to reduce printing:
    >>> np.random.seed(123)
    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> _parse_einsum_input(('...a,...a->...', a, b))
    ('za,xza', 'xz', [a, b]) # may vary
    >>> _parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    ('za,xza', 'xz', [a, b]) # may vary
    """

    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = [v for v in operands[1:]] # modified from Numpy's version, which wraps the first v in asanyarray(v)

        # Ensure all characters are valid
        for s in subscripts:
            if s in '.,->':
                continue
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)
                    
    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        unused = list(einsum_symbols_set - set(used))
        ellipse_inds = "".join(unused)
        longest = 0

        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(',')
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operands[num].ndim, 1)
                    ellipse_count -= (len(sub) - 3)

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace('...', '')
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = sub.replace('...', rep_inds)

        subscripts = ",".join(split_subscripts)
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = ""
            tmp_subscripts = subscripts.replace(",", "")
            for s in sorted(set(tmp_subscripts)):
                if s not in (einsum_symbols):
                    raise ValueError("Character %s is not a valid symbol." % s)
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = ''.join(sorted(set(output_subscript) -
                                         set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if char not in input_subscripts:
            raise ValueError("Output character %s did not appear in the input"
                             % char)

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the "
                         "number of operands.")

    return input_subscripts, output_subscript, operands