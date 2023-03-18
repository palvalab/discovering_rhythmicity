import numpy as np
import cupy as cp

from numpy.lib.stride_tricks import as_strided as np_as_strided
from cupy.lib.stride_tricks import as_strided as cp_as_strided

def crosscorrelation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.
    """
    py = np.pad(y.conj(), 2*maxlag, mode='constant')

    T = np_as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)

def crosscorrelation_cupy(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.
    """
    py = cp.pad(y.conj(), 2*maxlag, mode='constant')

    T = cp_as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)


def pad_with_lags(arr, lags, module=np):    
    pad_size = module.max(lags)
        
    arr_pad = module.pad(arr, int(pad_size), mode='constant')
    return arr_pad[(lags + pad_size)[:,None] + module.arange(arr.shape[0])]

def cc_lags_raw(x, y, lags, module=np):   
    T = pad_with_lags(x, lags, module=module)
    
    return T.dot(y)

def _cc_lags_normalized_internal(x, y, lags, module=np):   
    x_norm = x - x.mean()
    y_norm = y - y.mean()
    
    res = cc_lags_raw(x_norm, y_norm, lags, module=module)
    res /= (x.shape[0] * x.std() * y.std())
    
    return res

def cc_lags_normalized(x, y, lags, cuda=True):
    if cuda:
        x_gpu = cp.array(x)
        y_gpu = cp.array(y)
        lags_gpu = cp.array(lags)
        
        return cp.asnumpy(_cc_lags_normalized_internal(x_gpu, y_gpu, lags_gpu, module=cp))
    else:
        return _cc_lags_normalized_internal(x,y,lags, module=np)

def _compute_nm_pac_internal(sig_n: np.ndarray, sig_m: np.ndarray, lags: list, module=np):
    """
        Internal function, not for use! computes PAC for given signal and data
    :param sig: 1d array of analog (complex) data. Can be either numpy (CPU) or cupy (GPU).
    :param lags: vector of lags PAC should be computed with
    :param module: computational module that handles math operations (cupy or numpy)
    :return: array of PAC values. It has the same type as signal array
    """

    def _plv(x, y):
        return module.abs(module.mean(module.exp(module.complex(0, 1) * (x - y))))

    sig_n_angle = module.angle(sig_n)
    sig_m_angle = module.angle(sig_m)
    n_samples = sig_m.shape[0]

    res = module.zeros(len(lags))

    for idx, l in enumerate(lags):
        res[idx] = _plv(sig_n_angle, module.roll(sig_m_angle, l))

    return res


def _convert_phase(sig, m):
    ph = np.angle(sig)
    return np.exp(1j*(ph*m))


def _convert_phase_power(sig, m):
#     ph = np.angle(sig)
#     return np.exp(1j*(ph*m))
    return np.power(sig, m)


def compute_nm_pac(sig_n: np.ndarray, lags: list, m:int=2, cuda=True):
    """
        Computes PAC for given signal and data
    :param sig: 1d array of analog (complex) data.
    :param lags: vector of lags PAC should be computed with.
    :param cuda: indicates usage of GPU for computations.
    :return: array of PAC values.
    """
    
    sig_m = _convert_phase(sig_n, m)
    
    if cuda and HAS_CUPY:
        sig_n_gpu = cp.array(sig_n)
        sig_m_gpu = cp.array(sig_m)
        
        return cp.asnumpy(_compute_nm_pac_internal(sig_n_gpu, sig_m_gpu, lags, module=cp))
    else:
        return _compute_nm_pac_internal(sig_n, sig_m, lags, module=np)
    
def _compute_amplitude_correlation_internal(sig_n: np.array, sig_m: np.array, lags: list, module=np):
    """
        Internal function, not for use! computes PAC for given signal and data
    :param sig: 1d array of analog (complex) data. Can be either numpy (CPU) or cupy (GPU).
    :param lags: vector of lags PAC should be computed with
    :param module: computational module that handles math operations (cupy or numpy)
    :return: array of PAC values. It has the same type as signal array
    """

    def _plv(x, y):
        return module.abs(module.mean(module.exp(module.complex(0, 1) * (x - y))))

    n_samples = sig_n.shape[0]
    res = module.zeros(len(lags))

    for idx, l in enumerate(lags):
        res[idx] = module.corrcoef(sig_n, module.roll(sig_m, l))[0,1]

    return res
    
def compute_amplitude_correlation(sig_n: np.array, sig_m: np.array, lags: list, cuda=True):
    """
        Computes PAC for given signal and data
    :param sig: 1d array of analog (complex) data.
    :param lags: vector of lags PAC should be computed with.
    :param cuda: indicates usage of GPU for computations.
    :return: array of PAC values.
    """
    if cuda and HAS_CUPY:
        sig_n_gpu = cp.array(sig_n)
        sig_m_gpu = cp.array(sig_m)
        return cp.asnumpy(_compute_amplitude_correlation_internal(sig_n_gpu, sig_m_gpu, lags, module=cp))
    else:
        return _compute_amplitude_correlation_internal(sig_n, sig_m, lags, module=np)