import numpy as np
import scipy as sp

from collections import defaultdict

from .stats import interpolate_linear


def smooth_with_kde(samples, n_samples=100, bw=1, kde_x=None):
    kde_mdl = sp.stats.gaussian_kde(samples, bw_method=bw)
    
    if kde_x is None:
        kde_x = np.linspace(min(samples), max(samples), n_samples)
    kde_y = kde_mdl(kde_x)
    return kde_x, kde_y

def find_peaks_in_distribution_kde(samples, kde_x=None):
    if len(samples) == 0:
        return np.array([np.nan])
    elif len(samples) == 1:
        return np.array([samples[0]])   
    

    kde_x, kde_y = smooth_with_kde(samples, bw=None, kde_x=kde_x)
    peak_indices = sp.signal.find_peaks(kde_y, prominence=1e-3)[0]
    
    if len(peak_indices) == 0:
        return np.array([np.median(samples)])
    
    return kde_x[peak_indices]

def find_peaks_in_distribution(samples, **kwargs):
    try:
        return find_peaks_in_distribution_kde(samples, **kwargs)
    except:
        return np.array([np.median(samples)])


def detect_anatomy_peaks(channelwise_lengths, data_parcels, noise_level, f_vals, n_cortical=400):
    n_peaks_list = list()
    n_peaks_subjects = defaultdict(list)

    peak_freq_subjects = np.empty(n_cortical+93, dtype=object)
    peak_freq_subjects[:] = [list() for _ in range(n_cortical+93)]

    peak_freq_subjects_flat = np.empty(n_cortical+93, dtype=object)
    peak_freq_subjects_flat[:] = [list() for _ in range(n_cortical+93)]

    peak_freq_orig = np.empty((len(data_parcels), n_cortical+93), dtype=object)
    for i in range(len(data_parcels)):
        for j in range(n_cortical+93):
            peak_freq_orig[i,j] = list()

    cortical_peak_map = np.zeros((6,n_cortical + 93))

    for subj_idx, (subj_res, subj_mop) in enumerate(zip(channelwise_lengths, data_parcels)): 
        for spec_idx, (spectrum, mop) in enumerate(zip(subj_res.T, subj_mop)):
            spectrum_corr = spectrum/noise_level
            fnew, spectrum_interp = interpolate_linear(f_vals, spectrum_corr)
            
            sign_mask = (spectrum_interp >= 1)
            peak_indices = sp.signal.find_peaks(spectrum_interp, width=10, distance=30, height=1, prominence=0.25)[0]
            
            peak_indices_corrected = list()
            
            for idx in peak_indices:
                if all(sign_mask[idx-1:idx+1]):
                    peak_indices_corrected.append(idx)

            n_peaks = len(peak_indices)
            n_peaks_list.append(n_peaks)
            
            if mop >= 0 and n_peaks < 6:
                cortical_peak_map[n_peaks, mop] += 1
                
            if n_peaks == 10:
                raise RuntimeError('wtf')
                
            n_peaks_subjects[subj_idx].append(n_peaks)
            
            if mop >= 0 and 0 < n_peaks < 6:
                peak_freq_subjects[mop].append((fnew[peak_indices], spectrum_interp[peak_indices]))
    
    return peak_freq_subjects, cortical_peak_map, n_peaks_subjects


def build_peaks_map(peak_frequencies_by_subject, counter_known, alpha_band=(0,12), beta_band=(12,30), n_cortical=400):
    single_peaks_map = np.full((n_cortical, 2), np.nan)
    single_peaks_map_counter = np.full((n_cortical, 2), 0.0)
    single_peaks_flat = list()

    multi_peaks_map = np.full((n_cortical, 2), np.nan)
    multi_peaks_map_counter = np.full((n_cortical, 2), 0.0)
    multi_peaks_flat = list()

    merged_peaks_map = np.full((n_cortical, 2), np.nan)
    merged_peaks_map_counter = np.full((n_cortical, 2), 0.0)
    merged_peaks_flat = list()

    cond_list = [lambda x: len(x[0]) == 1, lambda x: len(x[0]) > 1, lambda x: len(x[0]) > 0]

    for condition, (peaks_map, peaks_counter, peaks_flat) in zip(cond_list, [
                                                                    (single_peaks_map, single_peaks_map_counter, single_peaks_flat),
                                                                    (multi_peaks_map, multi_peaks_map_counter, multi_peaks_flat),
                                                                    (merged_peaks_map, merged_peaks_map_counter, merged_peaks_flat),
                                                                    ]):

        for parc_idx in range(400):
            parcel_peaks = peak_frequencies_by_subject[parc_idx]
            single_peaks_orig = [p[0] for p in parcel_peaks if condition(p)]

            total_contacts = len(parcel_peaks)

            if len(single_peaks_orig) > 0:
                single_peaks = np.concatenate(single_peaks_orig)
            else:
                single_peaks = np.array(single_peaks_orig)

            peaks_flat.extend(single_peaks)
            peak_freqs_all = find_peaks_in_distribution(single_peaks, kde_x=np.linspace(2,40,100))

            for band_idx, (l,r) in enumerate([alpha_band, beta_band]):
                peak_freqs = peak_freqs_all[(peak_freqs_all > l) & (peak_freqs_all < r)]

                single_peaks_band = single_peaks[(single_peaks > l) & (single_peaks < r)]

                contacts_band = np.sum([((p > l) & (p < r)).any() for p in single_peaks_orig  ])
                frac = contacts_band / counter_known[parc_idx] if (counter_known[parc_idx] > 0) else 0.0
                peaks_counter[parc_idx, band_idx] = frac
                
                            
                if len(peak_freqs) == 0 and len(single_peaks_band) > 0:
                    peaks_map[parc_idx, band_idx] = np.median(single_peaks_band)
                elif len(peak_freqs) == 1:
                    peaks_map[parc_idx, band_idx] = peak_freqs[0]
                elif len(peak_freqs) > 1:
                    peaks_map[parc_idx, band_idx] = np.median(peak_freqs)

    res = ((single_peaks_map, single_peaks_map_counter, single_peaks_flat),
            (multi_peaks_map, multi_peaks_map_counter, multi_peaks_flat),
            (merged_peaks_map, merged_peaks_map_counter, merged_peaks_flat))

    return res    
    
