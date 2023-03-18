import numpy as np

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


def morph_burst_heatmap(mop_assignment, values, lengths, noise_level, n_parcels, functor=None):
    def _filter_unk(pair):
        return pair[0] != -1

    n_freqs = values.shape[0]
    
    res = np.zeros((n_parcels, n_freqs))
    counter = res.copy()
    
    for (parcel, spectrum, pacf_spectrum) in filter(_filter_unk, zip(mop_assignment, values.T.copy(), lengths.T.copy())):
        for freq_idx in np.where(pacf_spectrum >= noise_level)[0]:
            if (functor is None):
                res[parcel, freq_idx] += spectrum[freq_idx]
            else:
                res[parcel, freq_idx] += functor(spectrum[freq_idx])
                
            counter[parcel, freq_idx] += 1

    res /= counter
    
    return res

def morph_cohort_burst_heatmap(cohort_mops, cohort_values, cohort_lengths, noise_level, n_parcels, functor=None):
    n_subjs = len(cohort_mops)
    n_freqs = cohort_values[0].shape[0]
    
    res = np.zeros((n_subjs, n_parcels, n_freqs))
    
    for i in range(n_subjs):
        res[i] = morph_burst_heatmap(cohort_mops[i], cohort_values[i], cohort_lengths[i], noise_level, n_parcels, functor=functor)
    
    return res