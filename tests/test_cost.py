import numpy as np
import loupe


def test_sserror_forward():
    data = np.ones((10,10))
    model = loupe.ones((10,10))
    err = loupe.sserror(model, data, mask=None, gain_bias_invariant=False)
    assert err.data == 0 

    model = 0.9*loupe.ones((10,10)) 
    err = loupe.sserror(model, data, mask=None, gain_bias_invariant=False)
    assert np.allclose(err.data, 0.01)


def test_sserror_fit_gain():
    model = loupe.rand(size=(10,10))
    
    gain = np.random.uniform(low=-0.5, high=0.5)
    data = model.data.copy() * gain

    _model = loupe.sserror._G(model.data, mask=None)
    _data = loupe.sserror._G(data, mask=None)
    gain_est = loupe.sserror._fit_gain(_data, _model, mask=None)

    assert np.allclose(gain, gain_est)


def test_sserror_fit_bias():
    model = loupe.rand(size=(10,10))
    
    bias = np.random.uniform(low=-5, high=5)
    data = model.data.copy() + bias

    bias_est = loupe.sserror._fit_bias(data, model.data, mask=None, gain=1)

    assert np.allclose(bias, bias_est)


def test_sserror_fit_gain_bias():
    model = loupe.rand(size=(10,10))
    
    gain = np.random.uniform(low=-0.5, high=0.5)
    bias = np.random.uniform(low=-5, high=5)
    data = (model.data.copy() * gain) + bias

    _model = loupe.sserror._G(model.data, mask=None)
    _data = loupe.sserror._G(data, mask=None)
    gain_est = loupe.sserror._fit_gain(_data, _model, mask=None)
    bias_est = loupe.sserror._fit_bias(data, model.data, mask=None, gain=gain_est)

    assert np.allclose(gain, gain_est)
    assert np.allclose(bias, bias_est)


def test_sserror_forward_multi():
    model = loupe.rand(size=(3,10,10))
    gain = np.random.uniform(low=-0.5, high=0.5)
    bias = np.random.uniform(low=-5, high=5)
    data_clean = model.data.copy()
    data_clean += np.random.uniform(low=-.1, high=.1, size=(3,10,10))
    data = gain * data_clean + bias
    err = loupe.sserror(model, data, mask=None, gain_bias_invariant=True)

    assert err.data


def test_sserror_backward_multi():
    model = loupe.rand(size=(3,10,10), requires_grad=True)
    gain = np.random.uniform(low=-0.5, high=0.5)
    bias = np.random.uniform(low=-5, high=5)
    data_clean = model.data.copy()
    data_clean += np.random.uniform(low=-.1, high=.1, size=(3,10,10))
    data = gain * data_clean + bias
    err = loupe.sserror(model, data, mask=None, gain_bias_invariant=True)
    err.backward(grad=1.0)
    assert np.all(model.grad != 0)
    assert model.grad.shape == model.shape


def test_sserror_residual():
    model = loupe.rand(size=(3,10,10))
    gain = np.random.uniform(low=-0.5, high=0.5)
    bias = np.random.uniform(low=-5, high=5)
    data_clean = model.data.copy()
    data_clean += np.random.uniform(low=-.1, high=.1, size=(3,10,10))
    data = gain * data_clean + bias
    err = loupe.sserror(model, data, mask=None, gain_bias_invariant=False)

    r = err.residual()
    rr = model.data - data

    assert(np.allclose(r, rr))


def test_sserror_residual_gbi():
    model = loupe.rand(size=(3,10,10))
    gain = np.random.uniform(low=-0.5, high=0.5)
    bias = np.random.uniform(low=-5, high=5)
    data_clean = model.data.copy()
    data_clean += np.random.uniform(low=-.1, high=.1, size=(3,10,10))
    data = gain * data_clean + bias
    err = loupe.sserror(model, data, mask=None, gain_bias_invariant=True)

    r = err.residual()

    _model = loupe.sserror._G(model.data, mask=None)
    _data = loupe.sserror._G(data, mask=None)
    alpha = loupe.sserror._fit_gain(_data, _model, mask=None)
    rr = np.einsum('ijk,i->ijk', _model, alpha) - _data

    assert(np.allclose(r, rr))


def test_sserror_residual_mask():
    model = loupe.rand(size=(3,10,10))
    gain = np.random.uniform(low=-0.5, high=0.5)
    bias = np.random.uniform(low=-5, high=5)
    data_clean = model.data.copy()
    data_clean += np.random.uniform(low=-.1, high=.1, size=(3,10,10))
    data = gain * data_clean + bias
    mask = np.ones(shape=(3,10,10))
    mask[:,5,5] = 0
    mask[1,6,6] = 0

    err = loupe.sserror(model, data, mask, gain_bias_invariant=False)

    r = err.residual()
    rr = (model.data - data)*mask

    assert(np.allclose(r, rr))


def test_sserror_residual_gbi_mask():
    model = loupe.rand(size=(3,10,10))
    gain = np.random.uniform(low=-0.5, high=0.5)
    bias = np.random.uniform(low=-5, high=5)
    data_clean = model.data.copy()
    data_clean += np.random.uniform(low=-.1, high=.1, size=(3,10,10))
    data = gain * data_clean + bias
    mask = np.ones(shape=(3,10,10))
    mask[:,5,5] = 0
    mask[1,6,6] = 0
    
    err = loupe.sserror(model, data, mask, gain_bias_invariant=True)

    r = err.residual()

    _model = loupe.sserror._G(model.data, mask)
    _data = loupe.sserror._G(data, mask)
    alpha = loupe.sserror._fit_gain(_data, _model, mask)
    rr = mask * (np.einsum('ijk,i->ijk', _model, alpha) - _data)

    assert(np.allclose(r, rr))