import numpy as np

from .pac import morph_electrode_counter

def acf_mass_ratio(arr, threshold=2.42):
    mask = (arr >= threshold)
    indices = np.where(np.diff(mask))[0]

    start_indices = indices[::2]
    end_indices = indices[1::2]
    
    max_len = min(len(start_indices), len(end_indices))
    
    if max_len == 0:
        return np.nan
    
    best_idx = np.argmax(end_indices[:max_len] - start_indices[:max_len])
    start = start_indices[best_idx]
    end = end_indices[best_idx]

    if (end - start < 5):
        return np.nan
    
    sector = arr[start:end+1].copy()
    sector /= sector.sum()
    sector_sum = np.cumsum(sector)

    e_values = [np.abs(sector_sum - i/4).argmin()/10 for i in range(4)]
    
    return (e_values[3] - e_values[2])/(e_values[2] - e_values[1] + 1e-10)

def acf_magnitude(arr):
    arr_normed = arr - 1
    arr_normed[arr_normed < 0] = np.nan
    return np.nansum(arr_normed)

import numba

@numba.jit
def acf_mass_kurtosis(arr, threshold=2.42):
    mask = (arr >= threshold)
    indices = np.where(np.diff(mask))[0]

    start_indices = indices[::2]
    end_indices = indices[1::2]
    
    max_len = min(len(start_indices), len(end_indices))
    
    if max_len == 0:
        return np.nan
    
    best_idx = np.argmax(end_indices[:max_len] - start_indices[:max_len])
    start = start_indices[best_idx]
    end = end_indices[best_idx]

    if (end - start < 5):
        return np.nan
    
    sector = arr[start:end+1].copy()
    sector /= sector.sum()
    sector_sum = np.cumsum(sector)

    e_values = [np.abs(sector_sum - i/4).argmin()/10 for i in range(4)]
    
    return (e_values[3] - e_values[2])/(e_values[2] - e_values[1] + 1e-10)


def find_longest_segment(arr, min_length=5):
    indices = np.where(np.diff(arr))[0]

    start_indices = indices[::2]
    end_indices = indices[1::2]
    
    max_len = min(len(start_indices), len(end_indices))
    
    if max_len == 0:
        return None
    
    best_idx = np.argmax(end_indices[:max_len] - start_indices[:max_len])
    start = start_indices[best_idx]
    end = end_indices[best_idx]

    if (end - start < min_length):
        return None
    
    return np.arange(start, end + 1)

def compute_e_values(arr, threshold, min_length=10, percentiles=np.array([0.25, 0.5, 0.75])):
    mask = (arr >= threshold)

    segment_indices = find_longest_segment(mask, min_length=min_length)

    if (segment_indices is None):
        return None

    segment = arr[segment_indices].copy()
    segment /= segment.sum()
    segment_sum = np.cumsum(segment)

    e_values = np.abs(segment_sum.reshape(1,-1) - percentiles.reshape(-1,1)).argmin(axis=-1)

    return e_values

def acf_mass_ratio(arr, percentiles=np.array([0.25, 0.5, 0.75]), threshold=2.42, min_length=10):
    e_values = compute_e_values(arr, threshold=threshold, min_length=min_length, percentiles=percentiles)

    if (e_values is None):
        return np.nan

    return (e_values[2] - e_values[1])/(e_values[1] - e_values[0])


def morph_burst_heatmap(mop_assignment, values, n_parcels, functor=None):
    def _filter_unk(pair):
        return pair[0] != -1

    n_freqs = values.shape[0]
    
    counter = morph_electrode_counter(mop_assignment, n_parcels)
    counter = np.tile(counter, (n_freqs, 1)).T
    
    res = np.zeros((n_parcels, n_freqs))
    
    for (parcel, spectrum) in filter(_filter_unk, zip(mop_assignment, values.T.copy())):
        if functor:
            spectrum = functor(spectrum)
        
        nan_mask = np.isnan(spectrum)
        finite_mask = ~nan_mask

        res[parcel, finite_mask] += spectrum[finite_mask]
        counter[parcel, nan_mask] -= 1

    res /= counter
    
    return res

def morph_cohort_burst_heatmap(cohort_mops, cohort_values, n_parcels, functor=None):
    n_subjs = len(cohort_mops)
    n_freqs = cohort_values[0].shape[0]
    
    res = np.zeros((n_subjs, n_parcels, n_freqs))
    
    for i in range(n_subjs):
        res[i] = morph_burst_heatmap(cohort_mops[i], cohort_values[i], n_parcels, functor=functor)
    
    return res
