import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except:
    HAS_CUPY = False       
    

def _get_module(arr, force_gpu=True):
    if force_gpu:
        if HAS_CUPY:
            xp = cp
        else:
            raise RuntimeError('"force_gpu=True" while cupy is not installed')
    else:
        if HAS_CUPY:
            xp = cp.get_array_module(arr)
        else:
            xp = np
    
    return xp

def _dfa_boxcar(data_orig, win_lengths, xp):
    data = xp.array(data_orig, copy=True)
    win_arr = xp.array(win_lengths)
    
    data -= data.mean(axis=1, keepdims=True)
    data_fft = xp.fft.fft(data)

    n_chans, n_ts = data.shape
    is_odd = n_ts % 2 == 1

    n_windows = len(win_lengths)

    nx = (n_ts + 1)//2 if is_odd else n_ts//2 + 1
    data_power = 2*xp.abs(data_fft[:, 1:nx])**2

    if is_odd == False:
        data_power[:,~0] /= 2
        
    ff = xp.arange(1, nx)
    g_sin = xp.sin(xp.pi*ff/n_ts)
    
    hsin = xp.sin(xp.pi*xp.outer(win_arr, ff)/n_ts)
    hcos = xp.cos(xp.pi*xp.outer(win_arr, ff)/n_ts)

    hx = 1 - hsin/xp.outer(win_arr, g_sin)
    h = (hx / (2*g_sin.reshape(1, -1)))**2

    f2 = xp.inner(data_power, h)

    fluct = xp.sqrt(f2)/n_ts

    hy = -hx*(hcos*xp.pi*ff/n_ts - hsin/win_arr.reshape(-1,1)) / xp.outer(win_arr, g_sin)
    h3 = hy/(4*g_sin**2)

    slope = xp.inner(data_power, h3) / f2*win_arr
    
    return fluct, slope


def dfa_fft(data_orig, win_lengths, force_gpu=False, method='boxcar'):    
    module = _get_module(data_orig, force_gpu)

    allowed_methods = ('boxcar', )
    if not(method in allowed_methods):
        raise RuntimeError('Method {} is not allowed! Only {} are available'.format(method, ','.join(allowed_methods)))

    if method == 'boxcar':
        fluct, slope =  _dfa_boxcar(data_orig, win_lengths, xp=module)
        
    if not(module is np):
        fluct = module.asnumpy(fluct)
        slope = module.asnumpy(slope)
    
    dfa_values, residuals = np.polyfit(np.log2(win_lengths), np.log2(fluct.T), 1)
    
    return fluct, slope, dfa_values, residuals
