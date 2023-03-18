import scipy as sp
import scipy.stats

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import numpy as np
import pandas as pd
import pingouin as pg # need for pcorr

from sklearn.metrics import confusion_matrix

try:
    import cupy as cp
    HAS_CUPY = True
except:
    HAS_CUPY = False

from joblib import delayed, Parallel

def _get_func(module, func_name):
    try:
        func_object = getattr(module, func_name)
    except AttributeError as err:
        raise RuntimeError(f'There is no function called {func_name} in {module}!')
       
    return func_object

def get_module(arr):
    if HAS_CUPY:
        return cp.get_array_module(arr)
    else:
        return np

def interpolate(x,y, use_log=False, x_step=1e-2, smooth=True, filter_size=1.0):
    xvals = np.log(x) if use_log else x
    
    s = 0.25 if smooth else None
    f = sp.interpolate.splrep(xvals, y, k=3, s=s)
    
    xnew = np.arange(x[0], x[~0], step=x_step)
    xnew = np.log(xnew) if use_log else xnew
    ynew = sp.interpolate.splev(xnew, f)
        
    return xnew, ynew

def interpolate_linear(x, y):
    f = interp1d(x, y, kind='linear')
    xnew = np.linspace(x[0], x[~0], num=1000, endpoint=True)
    
    snew = f(xnew)
    snew = savgol_filter(snew, 11, 2)
    
    return xnew, snew

def interpolate_matrix(arr, frequencies, smooth=True):
    res = list()

    for i in range(arr.shape[0]):
        f_vals_interp, interpolated = interpolate(frequencies, arr[i], smooth=smooth)
        res.append(interpolated)
    
    res = np.array(res)
    return f_vals_interp, res

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.nanmean(a), sp.stats.sem(a, nan_policy='omit')
    h = se * sp.stats.t.ppf((1 + confidence) / 2., n-1)

    return m, m-h, m+h


def bootstrap_statistic(arr, n_rounds=1000, axis=0, func='nanmean'):
    xp = get_module(arr)
    func_object = _get_func(np, func) if isinstance(func, str) else func

    target_shape = list(arr.shape)
    reduce_shape = target_shape[axis]
    del target_shape[axis]
    target_shape = (n_rounds, *target_shape)

    res = xp.zeros(target_shape)
    idx = xp.arange(reduce_shape)
    
    for i in range(n_rounds):
        round_idx = xp.random.choice(idx, size=reduce_shape)
        
        res[i] = func_object(xp.take(arr, round_idx, axis=axis), axis=axis)
        
    return res

def bootstrap_n_peaks(n_peaks_cohort, mop_cohort, rounds=1000, n_cortical=400):
    subject_indices = np.arange(len(mop_cohort))
    
    res = np.zeros((rounds, 6, n_cortical + 93))
    
    for i in range(rounds):
        round_indices = np.random.choice(subject_indices, size=len(subject_indices))
        
        for subj_idx in round_indices:
            for n_peaks, mop in zip(n_peaks_cohort[subj_idx], mop_cohort[subj_idx]):
                
                if mop < 0 or n_peaks > 5:
                    continue
                
                res[i, n_peaks, mop] += 1
    
    return res


def subtract_alpha_level(mask, values, alpha):
    n_to_remove = int(len(mask)*alpha)

    res = mask.copy()
    
    sign_values = values[mask]
    
    if len(sign_values) == 0:
        return res
    
    sign_values = np.sort(sign_values)
    
    sign_threshold = sign_values[min(n_to_remove-1, len(sign_values)-1)]
    res[values < sign_threshold] = False
    
    return res

def corr_distance(u, v):
    mask = np.isfinite(u) & np.isfinite(v)

    u_finite = u[mask]
    v_finite = v[mask]
    
    corr_coeff = sp.stats.pearsonr(u_finite, v_finite)[0]
    
    return np.sqrt(1/2*(1 - corr_coeff))

def wm(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def wcov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - wm(x, w)) * (y - wm(y, w))) / np.sum(w)

def wcorr(x, y, w):
    """Weighted Correlation"""
    w_scaled = w / np.sum(w)
    return wcov(x, y, w_scaled) / np.sqrt(wcov(x, x, w_scaled) * wcov(y, y, w_scaled))

def nanfunc(func):
    def wrapper(*args, **kwargs):
        mask = np.logical_and.reduce([np.isfinite(arr) for arr in args])
        args_fixed = [arr[mask] for arr in args]

        return func(*args_fixed, **kwargs)
    
    return wrapper


def pvalue_from_corr(values, N):
    ab = N/2 - 1
    
    return 2*sp.special.btdtr(ab, ab, 0.5*(1 - np.abs(np.float64(values))))

def cov_along_axis(x, y, axis=0, ddof=1):
    xp = cp.get_array_module(x)
    
    n = x.shape[axis]
    
    xm = x.mean(axis=axis, keepdims=True)
    ym = y.mean()
    
    n_missing_dims = len(x.shape) - len(y.shape)
    y_rs = xp.expand_dims(y, tuple(np.arange(n_missing_dims) + 1))

    return xp.sum((x - xm)*(y_rs - ym), axis=axis)/(n-ddof)

def corr_along_axis(x, y, axis=0, ddof=1):
    xp = cp.get_array_module(x)
    
    n_missing_dims = len(x.shape) - len(y.shape)
    y_rs = xp.expand_dims(y, tuple(np.arange(n_missing_dims) + 1))
    
    cov_values = cov_along_axis(x,y, axis=axis, ddof=ddof)
    
    n = x.shape[axis]
    x_self_cov = xp.sum((x - x.mean(axis=axis, keepdims=True))**2, axis=axis) / (n - ddof)
    y_self_cov = xp.sum((y - y.mean())**2) / (n - ddof)
    
    div = xp.sqrt(x_self_cov*y_self_cov)
    
    return cov_values / div


def moving_average_fast(x, window_size):
    xp = get_module(x)

    is_odd = (window_size % 2) == 1
    w_half = window_size // 2

    start = w_half + 1
    end = -w_half + 1

    tail_size = window_size
    tail_start = w_half - 1
    tail_end = -tail_size - 1

    if is_odd:
        tail_start += 1
        tail_end += 1
        end -= 1

    res = xp.zeros_like(x)

    x_cumsum = xp.cumsum(x, axis=-1)

    res[..., start:end] = (x_cumsum[..., tail_size:] - x_cumsum[..., :-tail_size])/window_size

    # fill head & tail
    res[..., :w_half+1] = x_cumsum[..., tail_start:tail_size]/window_size
    res[..., -w_half:] = (x_cumsum[..., ~0:] - x_cumsum[..., tail_end:-w_half-1])/window_size
    
    return res

nanpearson = nanfunc(sp.stats.pearsonr)
wnanpearson = nanfunc(wcorr)
nanspearmanr = nanfunc(sp.stats.spearmanr)


def compute_cd(x, y, axis=(0,-1)):
    nx = x.shape[0]
    ny = y.shape[0]
    
    dof = nx + ny - 2
    
    x_mean = x.mean(axis=axis)
    y_mean = y.mean(axis=axis)
    
    x_std = (nx-1)*np.std(x, ddof=1, axis=axis) ** 2
    y_std = (ny-1)*np.std(y, ddof=1, axis=axis) ** 2
    
#     print(x_mean, x_std, y_mean, y_std)
    
    return (x_mean - y_mean) / np.sqrt((  x_std + y_std) / dof)


def compute_correlations_partial_pearson(pacf_values_subjectwise, amp_values_subjectwise, psd_values_subjectwise):
    n_freqs = pacf_values_subjectwise[0].shape[0]
    n_subjs = len(pacf_values_subjectwise)

    correlations_pearson = np.zeros((n_subjs, 3, n_freqs))
    correlations_partial = np.zeros((n_subjs, 3, n_freqs))

    for subj_idx, (subj_foofs, subj_amps, subj_pacs) in enumerate(zip(psd_values_subjectwise, amp_values_subjectwise, pacf_values_subjectwise)):
        n_freqs = subj_foofs.shape[0]
        
        for freq_idx in range(n_freqs):
            correlations_pearson[subj_idx, 0, freq_idx] = sp.stats.pearsonr(subj_foofs[freq_idx], subj_amps[freq_idx])[0]
            correlations_pearson[subj_idx, 1, freq_idx] = sp.stats.pearsonr(subj_pacs[freq_idx], subj_foofs[freq_idx])[0]
            correlations_pearson[subj_idx, 2, freq_idx] = sp.stats.pearsonr(subj_pacs[freq_idx], subj_amps[freq_idx])[0]
            
            tmp = np.vstack([subj_foofs[freq_idx], subj_amps[freq_idx], subj_pacs[freq_idx]]).T
            pcorr = pd.DataFrame(tmp, columns=['PSD', 'Amp', 'pAC']).pcorr()

            correlations_partial[subj_idx, 0, freq_idx] = pcorr['PSD']['Amp']
            correlations_partial[subj_idx, 1, freq_idx] = pcorr['PSD']['pAC']
            correlations_partial[subj_idx, 2, freq_idx] = pcorr['Amp']['pAC']
    
    return correlations_pearson, correlations_partial


def compute_pac_plv_sens(cohort_pac_values, cohort_cplv_values, cohort_ref_masks, noise_level, plv_surr_level):
    n_subjs = len(cohort_pac_values)
    n_freqs = cohort_pac_values[0].shape[0]

    res_sens = np.zeros((n_subjs, n_freqs))
    res_spec = res_sens.copy()

    for subj_idx in range(n_subjs):
        subj_mask = np.triu(cohort_ref_masks[subj_idx], 1).astype(bool)

        for freq_idx in range(n_freqs):
            stat_sign = (cohort_pac_values[subj_idx][freq_idx] > noise_level[freq_idx])
            stat_sign_2d = ((stat_sign.reshape(-1,1) & stat_sign.reshape(1,-1)) > 0).astype(int)

            plv_sign = (np.abs(cohort_cplv_values[subj_idx][freq_idx]) > plv_surr_level[freq_idx]).astype(int)

            stat_sign_flat = stat_sign_2d[subj_mask]
            plv_sign_flat = plv_sign[subj_mask]

            cm_plv = confusion_matrix(plv_sign_flat, stat_sign_flat, labels=[0,1])

            if (cm_plv[1,1] + cm_plv[1,0]) > 0:
                # precision
                res_sens[subj_idx, freq_idx] = cm_plv[1,1] / (cm_plv[1,1] + cm_plv[1,0])

            if (cm_plv[1,1] + cm_plv[0,1]) > 0:
                # recall
                res_spec[subj_idx, freq_idx] = cm_plv[1,1] / (cm_plv[1,1] + cm_plv[0,1])


    return res_sens, res_spec


def bootstrap_pac_plv_sens(cohort_pac_values, cohort_cplv_values, cohort_ref_masks, noise_level, plv_surr_level, n_rounds=100, n_jobs=32, temp_folder=None):
    def _boot_func():
        round_indices = np.random.choice(indices, len(indices))

        round_pac_values = [cohort_pac_values[j] for j in round_indices]
        round_cplv_values = [cohort_cplv_values[j] for j in round_indices]
        round_ref_masks = [cohort_ref_masks[j] for j in round_indices]

        round_sens, round_spec = compute_pac_plv_sens(round_pac_values, round_cplv_values, round_ref_masks, noise_level, plv_surr_level)

        return round_sens.mean(axis=0), round_spec.mean(axis=0)

    indices = np.arange(len(cohort_pac_values))

    if n_jobs == 1:
        round_values = [_boot_func() for i in range(n_rounds)]
    else:
        round_values = Parallel(n_jobs=n_jobs, temp_folder=temp_folder)(delayed(_boot_func)() for i in range(n_rounds))

    res_sens, res_spec = np.array(round_values).transpose(1,0,2)

    return res_sens


def compute_subject_binned_plv(cplv_values, pac_values, ref_mask, bins):
    n_bins =  bins.shape[0] - 1
    n_chans = cplv_values.shape[0]
    
    res = np.zeros((n_bins, n_bins))
    counter = np.zeros((n_bins, n_bins))
    
    pac_digitized = np.clip(np.digitize(pac_values, bins) - 1, 0, n_bins - 1)
    
    plv_indices = np.where(ref_mask)
    bin_indices = (pac_digitized[plv_indices[0]], pac_digitized[plv_indices[1]])

    np.add.at(res, bin_indices, np.abs(cplv_values[plv_indices]))
    np.add.at(counter, bin_indices, 1)
    
    res[counter != 0] /= counter[counter != 0]
    res[counter == 0] = np.nan

    return res, counter

def compute_cohort_binned_plv(cohort_cplv_values, cohort_pac_values, cohort_ref_masks=None, n_bins=10, pac_min=0.2, pac_max=0.5):
    n_subjs = len(cohort_cplv_values)
    n_freqs = cohort_cplv_values[0].shape[0]
    n_chans = cohort_cplv_values[0].shape[-1]
    
    pac_bins = np.linspace(pac_min, pac_max, n_bins+1)
    
    res = np.zeros((n_subjs, n_freqs, n_bins, n_bins ))
    counter = res.copy()
    
    pac_concat = np.concatenate(cohort_pac_values, axis=-1)
    
    cohort_pac_bins = np.zeros((n_subjs, n_freqs, n_bins+1))
    
    for subj_idx in range(n_subjs):
        if (cohort_ref_masks is None):
            subj_ref_mask = np.ones((n_chans, n_chans), dtype=bool)
        else:
            subj_ref_mask = cohort_ref_masks[subj_idx].copy()
        np.fill_diagonal(subj_ref_mask, False)
        
        for freq_idx in range(n_freqs):
            l,u = np.percentile(pac_concat[freq_idx], (1,99))
            pac_bins = np.linspace(l, u, n_bins + 1)
            
            res[subj_idx, freq_idx], counter[subj_idx, freq_idx] = compute_subject_binned_plv(cohort_cplv_values[subj_idx][freq_idx], 
                                                                 cohort_pac_values[subj_idx][freq_idx], 
                                                                 subj_ref_mask, bins=pac_bins)
            cohort_pac_bins[subj_idx, freq_idx] = pac_bins
    
    return (cohort_pac_bins[..., 1:] + cohort_pac_bins[..., :-1])/2, res



def compute_pac_sync_corr(pac_values, sync_values, ref_mask):
    n_freqs = pac_values.shape[0]
    
    res = np.zeros(n_freqs)

    for freq_idx in range(n_freqs):
        node_sync = np.average(sync_values[freq_idx], weights=ref_mask, axis=-1)
        res[freq_idx] = sp.stats.pearsonr(pac_values[freq_idx], node_sync)[0]
    
    return res
    

def compute_cohort_pac_sync_corr(cohort_pac_values, cohort_cplv_values, cohort_ref_masks):
    n_freqs = cohort_pac_values[0].shape[0]
    n_subjs = len(cohort_pac_values)

    res = np.zeros((n_subjs, n_freqs))

    for subj_idx in range(n_subjs):
        res[subj_idx] = compute_pac_sync_corr(cohort_pac_values[subj_idx], np.abs(cohort_cplv_values[subj_idx]), cohort_ref_masks[subj_idx])
    
    return res

def bootstrap_pac_sync_corr(cohort_pac_values, cohort_cplv_values, cohort_ref_masks, n_rounds=100, n_jobs=32, joblib_tmp_dir=None):
    def _boot_func():
        round_indices = np.random.choice(indices, len(indices))

        round_pac_values = [cohort_pac_values[j] for j in round_indices]
        round_cplv_values = [cohort_cplv_values[j] for j in round_indices]
        round_ref_masks = [cohort_ref_masks[j] for j in round_indices]

        round_values = compute_cohort_pac_sync_corr(round_pac_values, round_cplv_values, round_ref_masks)

        return round_values.mean(axis=0)

    indices = np.arange(len(cohort_pac_values))

    if n_jobs == 1:
        round_values = [_boot_func() for i in range(n_rounds)]
    else:
        round_values = Parallel(n_jobs=n_jobs, temp_folder=joblib_tmp_dir)(delayed(_boot_func)() for i in range(n_rounds))

    res = np.array(round_values)

    return res

def compute_surr_pac_sync_corr(cohort_pac_values, cohort_cplv_values, cohort_ref_masks, n_rounds=100, n_jobs=32, joblib_tmp_dir=None):
    def _boot_func():
        round_indices = np.random.choice(indices, len(indices))

        round_pac_values = [shuffle_along_axis(cohort_pac_values[j], axis=-1) for j in round_indices]
        round_cplv_values = [cohort_cplv_values[j] for j in round_indices]
        round_ref_masks = [cohort_ref_masks[j] for j in round_indices]

        round_values = compute_cohort_pac_sync_corr(round_pac_values, round_cplv_values, round_ref_masks)

        return round_values

    indices = np.arange(len(cohort_pac_values))

    if n_jobs == 1:
        round_values = [_boot_func() for i in range(n_rounds)]
    else:
        round_values = Parallel(n_jobs=n_jobs, temp_folder=joblib_tmp_dir)(delayed(_boot_func)() for i in range(n_rounds))

    res = np.array(round_values)

    return res

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def compute_pac_plv_sens_surr(cohort_pac_values, cohort_cplv_values, cohort_ref_masks, noise_level, plv_surr_level, n_rounds=100, n_jobs=1):
    def _boot_func():
        round_indices = np.random.choice(indices, size=len(indices))
        round_pac_values = [shuffle_along_axis(cohort_pac_values[idx], axis=0) for idx in round_indices]
        round_cplv_values = [cohort_cplv_values[idx] for idx in round_indices]
        round_ref_masks = [cohort_ref_masks[idx] for idx in round_indices]
        
        round_sens, round_spec = compute_pac_plv_sens(round_pac_values, round_cplv_values, round_ref_masks, noise_level, plv_surr_level)

        return round_sens

    indices = np.arange(len(cohort_pac_values))
    
    if n_jobs == 1:
        res_spec = [_boot_func() for i in range(n_rounds)]
    else:
        res_spec = Parallel(n_jobs=n_jobs)(delayed(_boot_func)() for i in range(n_rounds))

    return np.array(res_spec)

def manual_corr_along_axis(x, y, corr_func=sp.stats.pearsonr):
    res = np.zeros(x.shape[1:])
    res_pv = res.copy()
    for idx in np.ndindex(*x.shape[1:]):
        new_idx = (slice(None), )  +idx
        xv = x[new_idx]

        mask = np.isfinite(xv)

        res[idx], res_pv[idx] = corr_func(xv[mask], y[mask])

    return res, res_pv

def jackknife_correlation(x, y, corr_callable=sp.stats.pearsonr, **kwargs):
    n_subjs = x.shape[0]
    res_coeffs = np.zeros((n_subjs, *x.shape[1:]))
    res_pv = np.zeros((n_subjs, *x.shape[1:]))
    
    mask = np.ones(n_subjs, dtype=bool)
    mask[0] = False
    
    for i in range(n_subjs):
        
        res_coeffs[i], res_pv[i] = corr_callable(x[mask], y[mask], **kwargs)
        
        mask = np.roll(mask, 1)
    
    return res_coeffs, res_pv


#### MDD figure utils
def stratified_sample(df, by='gender', cohorts=['S','D'], index_col='index', n_rounds=1000):
    to_group = ['cohort', by]
    
    grouped = df.groupby(to_group)    
    min_count = grouped.size().min()
    
    for _ in range(n_rounds):
        data_sampled = grouped.apply(lambda x: x.sample(min_count))
        to_return = (data_sampled.loc[cohort_label, index_col].values for cohort_label in cohorts)
        yield to_return
        
def stratified_stats(data, df, cohorts=['S', 'D'], by='gender', index_col='index', n_rounds=1000):
    res = np.zeros((len(cohorts), n_rounds, *data.shape[1:]), dtype=data.dtype)
    
    for i, round_indices in enumerate(stratified_sample(df, cohorts=cohorts, by=by, index_col=index_col, n_rounds=n_rounds)):
        for cohort_idx, data_indices in enumerate(round_indices):
            res[cohort_idx, i] = data[data_indices].mean(axis=0)
    
    return res

def compute_null_diff_freqwise(data, n_left, n_rounds=1000):
    np.random.seed(42)
    res = np.zeros((n_rounds, data.shape[1]))
    
    data_copy = data.transpose(1,2,0).reshape(-1, data.shape[1])
    
    for i in range(n_rounds):
        np.random.shuffle(data_copy)
        
        diff = data_copy[:n_left].mean(axis=0) - data_copy[n_left:].mean(axis=0)
        res[i] = diff
    
    return res

def _compute_ranks(x, y):
    xr = np.zeros_like(x)

    for axis_idx in np.ndindex(x.shape[1:]):
        i,j = axis_idx
        xr[:, i, j] = sp.stats.rankdata(x[:, i, j])

    yr = sp.stats.rankdata(y)
    
    return xr, yr

def spearmanr_along_axis(x, y, axis=0, ddof=2, rank_method='average', ranks_given=False):
    if ranks_given:
        xr = x
        yr = y
    else:
        xr, yr = _compute_ranks(x, y)
    
    return corr_along_axis(xr, yr, ddof=ddof, axis=axis)


def compute_null_correlation_jk(x, y, n_rounds=1000, corr_func='pearsons'):
    corr_callable = corr_along_axis if (corr_func == 'pearsonr') else spearmanr_along_axis
    
    res = np.zeros((n_rounds, *x.shape[1:]))
    
    xr, yr = _compute_ranks(x, y)
    
    yr_shuf = yr.copy()
    
    for i in range(n_rounds):
        np.random.shuffle(yr_shuf)
        
        round_vals = corr_callable(xr, yr_shuf, ranks_given=True)
        res[i] = round_vals
        
    return res