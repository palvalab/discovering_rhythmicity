import numpy as np
import cupy as cp

import tqdm

from crosspy.preprocessing.signal import filter_data

def compute_amplitude_spectrum(data, frequencies, sampling_rate, lags_cycles, omega=7.5, disable_tqdm=False):
    n_frequencies = len(frequencies)
    n_channels = data.shape[0]
    n_lags = len(lags_cycles)
    
    res = np.zeros((n_frequencies, n_channels))
    
    for freq_idx, freq in enumerate(tqdm.tqdm(frequencies, disable=disable_tqdm)):
        data_complex = filter_data(data, sampling_rate, freq, omega=omega, n_jobs='cuda')
        res[freq_idx] = cp.abs(data_complex).mean(axis=-1).get()
        
    return res

def create_oscillation(data_base, frequency, omega=5, sr=1000, snr=3):
    res = data_base.copy()
    
    res += filter_data(data_base, sr, frequency, omega, n_jobs='cuda').real.get()*snr
    
    return res