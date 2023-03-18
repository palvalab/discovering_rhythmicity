import numpy as np

from joblib import Parallel, delayed

from .stats import interpolate_linear

import fooof


def fit_spectrum(spectrum, f_vals, min_height=0.5):
    f = fooof.FOOOF(min_peak_height=min_height, max_n_peaks=5, peak_threshold=2.0)

    f.freqs = f_vals.copy()
    f.freq_res = f_vals[1] - f_vals[0]
    f.freq_range = [f_vals[0], f_vals[~0]]
    f._spectrum_flat = spectrum.copy()

    g_fit = f._fit_peaks(spectrum)
    
    return g_fit

def fit_spectrum_interp(spectrum, f_vals, correction):
    min_height = np.std(spectrum - correction)*1.5
    
    xnew, snew = interpolate_linear(f_vals, spectrum - correction)

    g_fit = fit_spectrum(snew, xnew, min_height=min_height)
    
    return xnew, snew, g_fit

def get_foof_model(freqs, spectrum, f_min=2, f_max=100):
    fm = fooof.FOOOF(max_n_peaks=4)
    fm.fit(freqs, spectrum, [2, 98])
    
    return fm

def get_foofed_spectrum(freqs, spectrum, f_min=2, f_max=100, freqs_to_map=None, normalize=False):
    fm = get_foof_model(freqs, spectrum, f_min=f_min, f_max=f_max)
    res = fm._spectrum_flat
    
    if normalize:
        res /= -fm._ap_fit
    
    if (res is None):
        return np.zeros_like(freqs_to_map)
    
    if not(freqs_to_map is None):
        new_psd_freqs = freqs[(freqs >= 2) & (freqs <= 98)]
        
        psd_freq_indices = np.abs(new_psd_freqs[None] - freqs_to_map[:, None]).argmin(axis=-1)
        res = res[psd_freq_indices]

    return res

def get_foofed_spectrum_chanwise(freqs, spectrums, f_min=2, f_max=100, freqs_to_map=None, normalize=False):
    n_chans = spectrums.shape[-1]
    n_freqs = freqs.shape[0] if (freqs_to_map is None) else freqs_to_map.shape[0]
    
    res = np.array(Parallel(n_jobs=32)(delayed(get_foofed_spectrum)(freqs, spectrum, f_min=f_min, f_max=f_max, freqs_to_map=freqs_to_map, normalize=normalize) for spectrum in spectrums))
    
    return res.T