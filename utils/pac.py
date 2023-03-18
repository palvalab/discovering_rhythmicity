import numpy as np
import scipy as sp

try:
    import cupy as cp
    HAS_CUPY = True
except:
    HAS_CUPY = False

import itertools

import tqdm

from typing import Sequence

from crosspy.core.phase import compute_instantaneous_frequency
from crosspy.preprocessing.signal import filter_data

from .pac_gpu import _pac_kernel

def transform_to_cdf(data: np.ndarray, derivative_threshold: float=1e-3, min_consequent: int=3) -> np.ndarray:
    """
        Transforms an array of pAC values to cumulative density function
    :param data: 1d vector of pAC values
    :param derivative_threshold: threshold for the first derivative that considered as plateau
    :param min_consequent: minimum amount of samples <= threshold to be considered as plateau
    :return: 1d vector that represents CDF
    """
    xp = cp.get_array_module(data)

    diff = xp.abs(xp.diff(data))
    
    thresholded = xp.convolve((diff <= derivative_threshold).astype(int), xp.ones(min_consequent, dtype=int))
    baseline_idx = xp.argmax(thresholded)
    
    cdf = data.copy()
    cdf[baseline_idx:] = 0
    cdf /= cdf.sum()
    
    return xp.cumsum(cdf)

def get_length_by_cdf(data: np.ndarray, lag_values: Sequence[float], cdf_threshold: float=0.9, 
                        interpolate: bool=False, **kwargs) -> float:
    """
        Get pAC length. It is computed as the first lag with CDF >= threshold
    :param data: 1d vector of pAC values
    :param lag_values: sequence of lags
    :param cdf_threshold: threshold to compute length
    :param derivative_threshold: threshold to compute CDF
    :return: pAC length
    """

    xp = cp.get_array_module(data)

    cdf = transform_to_cdf(data, **kwargs)
    
    if interpolate:
        lags_interp = xp.linspace(0, lag_values[~0], 1000)
        cdf_interp = xp.interp(lags_interp, lag_values, cdf)
        
        length_idx = xp.argmax(cdf_interp >= cdf_threshold)

        return lags_interp[length_idx]
    
    length_idx = xp.argmax(cdf >= cdf_threshold)

    return lag_values[length_idx]


def kernel_numpy(data, data_conj, lag_samples):
    phase_diff = np.zeros(data.shape)

    for curr_row in range(data.shape[0]):
        for curr_col in range(data.shape[1]):
            lag = lag_samples[curr_row, curr_col]
            if lag + curr_col < data.shape[-1]:
                phase_diff[curr_row, curr_col] = data[curr_row, curr_col] * data_conj[curr_row, curr_col + lag]
            else:
                phase_diff[curr_row, curr_col] = np.nan
    
    return phase_diff


def compute_pac_with_lags(data, lag_cycles, signal_frequencies, sr=1000):
    pac_vals = cp.zeros((data.shape[0], len(lag_cycles)))
    
    data_conj = cp.conj(data)
    phase_diff = data.copy()
    
    for lag_idx, lag in enumerate(lag_cycles):
        lag_samples = cp.rint(lag*sr/signal_frequencies).astype(int)

        _pac_kernel(data, data_conj, lag_samples, data.shape[-1], phase_diff)
        lag_plv = cp.abs(cp.nanmean(phase_diff, axis=-1))
        pac_vals[:, lag_idx] = lag_plv

    return pac_vals.get()

def compute_pac_spectrum(data, frequencies, sampling_rate, lags_cycles, omega=7.5, disable_tqdm=False):
    n_frequencies = len(frequencies)
    n_channels = data.shape[0]
    n_lags = len(lags_cycles)
    
    res = np.zeros((n_frequencies, n_channels, n_lags))
    
    for freq_idx, freq in enumerate(tqdm.tqdm(frequencies, disable=disable_tqdm)):
        data_complex = filter_data(data, sampling_rate, freq, omega=omega, n_jobs='cuda')
        data_complex /= cp.abs(data_complex)

        data_if = compute_instantaneous_frequency(data_complex, sr=sampling_rate).mean(axis=-1)
        
        res[freq_idx] = compute_pac_with_lags(data_complex, lags_cycles, data_if, sr=sampling_rate)
        
    return res

def extract_pac_lengths(data_pac, lags_cycles, interp=False):
    def _process_profile(profile):
        return get_length_by_cdf(profile, lags_cycles)

    res = np.zeros(data_pac.shape[:-1])
    for freq_idx in range(data_pac.shape[0]):
        res[freq_idx] = [_process_profile(profile)  for profile in data_pac[freq_idx]]
        
    return res


def compute_noise_with_notch(noise_level, notch_manual=None):
    if notch_manual is None:
        notch_level = (noise_level[64] + noise_level[68])/2
    else:
        notch_level = notch_manual
        
    noise_updated = noise_level.copy()

    noise_updated[62:71] = notch_level
    noise_updated[79:] = notch_level
    
    return noise_updated


class IdentityFunctor:
    def __call__(self, x):
        return x
    
    
class PacSignificanceFunctor:
    def __init__(self, noise_pacs, significance_level=90, correct_notch=False):
        self.significance_level = significance_level
        
        noise_collapsed = np.concatenate(noise_pacs, axis=-1)
        self.noise_level = np.percentile(noise_collapsed, self.significance_level, axis=-1)
        
        if correct_notch:
            self.noise_level = compute_noise_with_notch(self.noise_level)
        
    def __call__(self, x):
        return x >= self.noise_level


def morph_pair_counter(mop_assignment, n_parcels):
    def _filter_unk(pair):
        return not((pair[0] == -1) or (pair[1] == -1))

    ix, iy = zip(*filter(_filter_unk, itertools.product(mop_assignment, mop_assignment)))
    res = np.zeros((n_parcels, n_parcels))
    
    np.add.at(res, (ix, iy), 1)
    res[res == 0] = np.nan
    
    return res

def morph_electrode_counter(mop_assignment, n_parcels):
    res = np.zeros(n_parcels)
    mop_filtered = [idx for idx in mop_assignment if (idx != -1)]
    
    np.add.at(res, mop_filtered, 1)
    res[res == 0] = np.nan
    
    return res

def morph_pac_heatmap(mop_assignment, values, n_parcels, functor=None):
    def _filter_unk(pair):
        return pair[0] != -1

    n_freqs = values.shape[0]
    
    counter = morph_electrode_counter(mop_assignment, n_parcels)
    res = np.zeros((n_parcels, n_freqs))
    
    for (parcel, spectrum) in filter(_filter_unk, zip(mop_assignment, values.T.copy())):
        if functor:
            spectrum = functor(spectrum)
        res[parcel] += spectrum

    res /= counter.reshape(-1,1)
    
    return res

def morph_cohort_pac_heatmap(cohort_mops, cohort_values, n_parcels, functor=IdentityFunctor()):
    n_subjs = len(cohort_mops)
    n_freqs = cohort_values[0].shape[0]
    
    res = np.zeros((n_subjs, n_parcels, n_freqs))
    
    for i in range(n_subjs):
        res[i] = morph_pac_heatmap(cohort_mops[i], cohort_values[i], n_parcels, functor=functor)
    
    return res




def smooth_adjacent_values(values, counter, adjacency_matrix):
    res = values.copy()
    
    for i in range(len(values)):        
        adj_indices = np.where(adjacency_matrix[i] & np.isfinite(values))[0]
        adj_counter = counter[adj_indices]
        res[i] = np.average(values[adj_indices], weights=adj_counter)
    
    return res

def fft_acf(a):
    xp = cp.get_array_module(a)
    pad_values = xp.zeros((*a.shape[:-1], a.shape[-1]-1))
    
    a_pad = xp.concatenate((a,pad_values), axis=-1) # added zeros to your signal
    A = xp.fft.fft(a_pad)
    S = xp.conj(A)*A
    c_fourier = xp.fft.ifft(S)
    
    c_fourier = c_fourier[..., :(c_fourier.shape[-1]//2)+1]
    plv_fourier = np.abs(c_fourier) / xp.arange(a.shape[-1], 0, -1).reshape(1,-1)
    
    return plv_fourier
