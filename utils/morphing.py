import numpy as np

def morph_plv(cplv_values, mop_values, ref_mask, n_parcels=293):
    res = np.zeros((cplv_values.shape[0], n_parcels, n_parcels))
    counter = res.copy()

    for i in range(cplv_values.shape[1]):
        for j in range(cplv_values.shape[1]):
            if ref_mask[i,j] == False:
                continue

            ip = mop_values[i]
            jp = mop_values[j]
            
            value = np.abs(cplv_values[:, i,j])
            
            if ip >= 0 and jp >= 0:
                res[:, ip, jp] += value
                counter[:, ip, jp] += 1
                
                if ip != jp:
                    res[:, jp, ip] += value
                    counter[:, jp, ip] += 1
                    
    
    res /= counter
    
    return res

def morph_dfa(dfa_values, mop_values, ref_mask, n_parcels=293):
    res = np.zeros((dfa_values.shape[0], n_parcels))
    counter = res.copy()
    
    for i in range(dfa_values.shape[1]):
        ip = mop_values[i]

        res[:, ip] += dfa_values[:, i]
        counter[:, ip] += 1                    
    
    res /= counter
    
    return res

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