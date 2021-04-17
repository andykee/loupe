import numpy as np
import pytest

import loupe


@pytest.mark.parametrize("subs, shapes",[
    # sums
    ('i->', ((5,),)),
    ('ij->', ((5,5),)),
    ('ijk->', ((5,5,5),)),

    # element-wise multiply
    ('i,i->i', ((3,),(3,))),
    ('ij,ij->ij', ((3,4),(3,4))),

    # inner
    ('i,i', ((3,),(3,))),
    ('ij,kj->ik', ((3,4),(5,4))),

    # outer
    ('i,j->ij', ((3,),(4,))),

    # dot
    ('ij,jk', ((3,4),(4,3))),
    
    # basis expansion
    ('ijk,i->jk', ((10,5,5),(10))), # w/sum
    ('ijk,i->ijk', ((10,5,5),(10))), # w/o sum
])
def test_einsum_forward(subs, shapes):
    inputs = [np.random.uniform(low=-1, high=1, size=shape) for shape in shapes]
    assert np.all(loupe.einsum(subs, *inputs).data == np.einsum(subs, *inputs))


@pytest.mark.parametrize("subs, shapes, func", [
    # element-wise multiply
    ('i,i->i', ((3,),(3,)), loupe.multiply), 
    ('ij,ij->ij', ((3,3),(3,3)), loupe.multiply)
])
def test_einsum_backward_multiply(subs, shapes, func):
    inputs = [loupe.rand(low=-1, high=1, size=shape, requires_grad=True) for shape in shapes]
    out = loupe.einsum(subs, *inputs)
    out.backward(np.ones_like(out))

    einsum_grads = []
    for x in inputs:
        einsum_grads.append(x.grad.copy())
        x.zero_grad()

    # repeat the calculation with an explicit loupe function
    del out
    out = func(*inputs)
    out.backward(np.ones_like(out))
    loupe_grads = []
    for x in inputs:
        loupe_grads.append(x.grad.copy()) 

    assert np.allclose(einsum_grads, loupe_grads)


#def test_einsum_backward_basis_expansion():
#    pass
