import os
import glob

import numpy as np
import scipy as sp

import pandas as pd

import mne

from matplotlib import pyplot as plt

from joblib import Parallel, delayed

import tqdm

from crosspy.preprocessing.seeg.support import clean_montage, drop_monopolar_channels
from crosspy.preprocessing.signal import filter_data
from crosspy.core.phase import compute_instantaneous_frequency

import cupy as cp

import pickle

import re

from bids import BIDSLayout

from scipy.optimize import curve_fit

#use 2070
# cp.cuda.Device(0).use()

import warnings
warnings.filterwarnings("ignore", "Channel names are not unique")

from utils.colornoise import powerlaw_psd_gaussian

def make_bipolar(data_fname, montage_filename):
    raw = mne.io.read_raw_edf(data_fname, preload=False, verbose=False)
    mne.rename_channels(raw.info, lambda name: re.sub(r'(POL|SEEG)\s+', '', name).strip())

    channel_types = dict()

    for ch in raw.ch_names:
        result = re.match(r'^[A-Z][\']?\d+', ch)
        if result:
            channel_types[ch] = 'seeg'

    raw.set_channel_types(channel_types)

    montage = pd.read_csv(montage_filename, delimiter='\t')
    montage.drop_duplicates(subset='name', inplace=True)

    anode,cathode = clean_montage(raw.ch_names, montage.anode.tolist(), montage.cathode.tolist())

    raw.load_data()

    bipo = mne.set_bipolar_reference(raw, list(anode), list(cathode), copy=True, verbose=False)
    bipo = drop_monopolar_channels(bipo)
    bipo.drop_channels(bipo.info['bads'])

    picks_seeg = mne.pick_types(bipo.info, meg=False, seeg=True)

    non_seeg_chans = [ch_name for ch_idx, ch_name in enumerate(bipo.ch_names) if not(ch_idx in picks_seeg)]
    bipo.drop_channels(non_seeg_chans)

    bipo.notch_filter(np.arange(50, bipo.info['sfreq']//2, 50), trans_bandwidth=0.1, verbose=False)

    return bipo


_pac_kernel = cp.ElementwiseKernel(
             'T x, raw T y, raw I lags, int32 n_cols', 'raw C output',
             '''
             int curr_col = i % n_cols;
             int curr_row = i / n_cols;
             
             int sample_lag = lags[curr_row];
             int offset_idx = curr_col + sample_lag;
             
             if(offset_idx < n_cols) {
                 int compare_idx = curr_row*n_cols + offset_idx;
                 output[i] = x * y[compare_idx];
             }
             else
             {
                 output[i] = __int_as_float(0xFFE00000);
             }
                                       
             ''',
             '_pac_kernel',
            )

def compute_pac_with_lags(data, lag_cycles, signal_frequencies, sr=1000):
    pac_vals = cp.zeros((data.shape[0], len(lag_cycles)))

    ns = data.shape[-1]
    
    data_conj = cp.conj(data)
    phase_diff = data.copy()
    
    for lag_idx, lag in enumerate(lag_cycles):
        lag_samples = cp.rint(lag*sr/signal_frequencies).astype(int)
        
        _pac_kernel(data, data_conj, lag_samples, data.shape[-1], phase_diff)
        pac_vals[:, lag_idx] = cp.abs(cp.nanmean(phase_diff, axis=-1))
    
    return cp.asnumpy(pac_vals)


def process_multiple_frequencies(data_orig, data_noise, frequencies, lags_cycles, sr, cuda_idx=0, use_tqdm=False):
    with cp.cuda.Device(cuda_idx):
        data_list = np.zeros((len(frequencies), data_orig.shape[0], len(lags_cycles)))
        noise_list = np.zeros((len(frequencies), data_orig.shape[0], len(lags_cycles)))

        n_chans = data_orig.shape[0]

        for (start, end) in [(0, n_chans//2),(n_chans//2, n_chans)]:
            data_gpu = cp.array(data_orig[start:end], dtype=cp.float32)
            noise_gpu = cp.array(data_noise[start:end], dtype=cp.float32)

            for freq_idx, f in enumerate(tqdm.tqdm(frequencies, leave=False, disable=not(use_tqdm))):            
                data_complex = filter_data(data_gpu, sr, frequency=f, omega=7.5, n_jobs='cuda')

                data_if = compute_instantaneous_frequency(data_complex, sr=1000).mean(axis=-1)

                data_complex /= cp.abs(data_complex)

                noise_complex = filter_data(noise_gpu, sr, frequency=f, omega=7.5, n_jobs='cuda')

                noise_if = compute_instantaneous_frequency(noise_complex, sr=1000).mean(axis=-1)

                noise_complex /= cp.abs(noise_complex)

                # lags_samples = np.rint(lags_cycles*(sr/f)).astype(int)

                data_pac = np.abs(compute_pac_with_lags(data_complex, lags_cycles, data_if))
                data_list[freq_idx, start:end] = data_pac

                noise_pac = np.abs(compute_pac_with_lags(noise_complex, lags_cycles, noise_if))
                noise_list[freq_idx, start:end] = noise_pac
            
    return (data_list, noise_list)

def create_noise(data):
    noise_arr = np.random.normal(size=data._data.shape)
    nchans = noise_arr.shape[0]

    ch_types = ['seeg']*nchans
    ch_names = ['noise #{}'.format(i) for i in range(nchans)]

    info = mne.create_info(ch_names=ch_names, sfreq=data.info['sfreq'], ch_types=ch_types)
    noise = mne.io.RawArray(noise_arr, info, verbose=False)

    noise.notch_filter(np.arange(50, data.info['sfreq']//2, 50), trans_bandwidth=0.1, verbose=False)

    return noise

def main():
    cutoff_secs = 10.0
    lags_cycles = np.arange(200)/10
 
    notch_target = [50, 150, 250, 350, 450]
    root_path = os.path.join('..', 'seeg_phases', 'data', 'SEEG_redux_BIDS')
    layout = BIDSLayout(root_path)

    f_vals = [2]

    while f_vals[~0] < 99:
        f_vals.append(f_vals[~0]*1.05)

    f_vals = np.array(f_vals)

    n_freqs = len(f_vals)

    for subject in tqdm.tqdm(layout.get(target='subject', extension='edf')): 
        res_fname = 'pac_sub-{}.pickle'.format(subject.entities['subject'])
        res_path = os.path.join('derivatives', 'pac_signal_frequency_7.5', res_fname)

        if os.path.exists(res_path):
            print('Subject {} is processed!'.format(subject.entities['subject']))
            continue

        montage_filename = os.path.join(subject.dirname,  'sub-{}_montage.tcsv'.format(subject.entities['subject']))
        data_filename = subject.path

        data_ref = make_bipolar(data_filename, montage_filename)
        noise = create_noise(data_ref)
        
        sfreq = data.info['sfreq']
        cutoff_samples = int(sfreq*cutoff_secs)

        pac_joblib = Parallel(n_jobs=2)(delayed(process_multiple_frequencies)(data_ref._data[:, cutoff_samples:], noise._data, job_freqs, lags_cycles, sfreq, cuda_idx) for
                        (job_freqs, cuda_idx) in zip([f_vals[:n_freqs//2], f_vals[n_freqs//2:]], [0,1]))
       
        pac_freqwise = np.concatenate([pac_joblib[0][0], pac_joblib[1][0]])
        noise_freqwise = np.concatenate([pac_joblib[0][1], pac_joblib[1][1]])

        res = {'data_pac': pac_freqwise, 'noise_pac': noise_freqwise, 'f_vals': f_vals}
        res['omega'] = 7.5

        pickle.dump(res, open(res_path, 'wb'))
    

if __name__ == '__main__':
    main()
