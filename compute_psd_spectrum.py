"""
@author: Vladislav Myrov
"""

import os

import mne

import re
import json
import pickle
import glob

import itertools
import argparse
import tqdm

import numpy as np
import pandas as pd
import cupy as cp
import scipy as sp
import scipy.signal

import cusignal

from bids import BIDSLayout

from crosspy.preprocessing.seeg.support import clean_montage, drop_monopolar_channels
from crosspy.preprocessing.seeg.seeg_utils import create_reference_mask, get_electrode_distance
from crosspy.preprocessing.signal import filter_data

from utils.io import load_mop, load_electrodes, load_montage
from utils.dfa import dfa_fft

np.random.seed(42)
cp.random.seed(42)

def is_bipolar(ch_name):
    a,c = ch_name.split('-')
    return len(a) > 0 and len(c) > 0

def create_bipolar_ref_mask(ch_names):
    n_channels = len(ch_names)
    
    mask = np.ones((n_channels, n_channels), dtype=int)

    for i,j in itertools.product(range(n_channels), range(n_channels)):
        ch1, ch2 = ch_names[i], ch_names[j]

        ch1 = ch1.split('-')
        ch2 = ch2.split('-')

        if len(set(ch1) & set(ch2)) > 0 or i == j :
            mask[i,j] = 0
    
    return mask

def make_bipolar(data_fname, montage_filename, lowpass_frequency=440):
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
    bipo.filter(None, lowpass_frequency, verbose=False, n_jobs=32)
    
    return bipo

def get_frequencies():
    f_vals = [2]

    while f_vals[~0] < 99:
        f_vals.append(f_vals[~0]*1.05)

    f_vals = np.array(f_vals)
    return f_vals


def get_ez_samples_mask(windows_data, data):
    mask = np.full(data.shape[1], fill_value=True)
    
    for start, end in windows_data[['Start', 'End']].values:
        mask[start:end] = False
    
    return mask

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

def fix_mop(mop, bipolar_mask):
    res = dict()

    res['mOp'] = mop['mOp'][bipolar_mask][:, bipolar_mask]
    res['parcel_assign'] = mop['parcel_assign'][bipolar_mask]
    res['channel_names'] = [name for (name, flag) in zip(mop['channel_names'], bipolar_mask) if flag]

    return res
 
def main():
    analysis_params = json.load(open('psd_spectrum_config.json'))

    data_root = os.path.join(analysis_params['data_path'])
    layout = BIDSLayout(data_root)

    frequencies = get_frequencies()

    bad_subjects = ['02', # weird mop
                    '06',
                    '40',
                    '42',
                    '57']
    old_bads = ('40', '06', '42', '60', '57', '18', '02', '19')
    # to_recompute = set(old_bads) - set(bad_subjects)
    to_recompute = ['60']

    for subject in layout.get(target='subject', extension='edf'): 
        subject_code = subject.entities['subject']
        result_fname = os.path.join(analysis_params['output_path'], 'sub-{}_spectrum.pickle'.format(subject.entities['subject'])) 
        
        # if os.path.exists(result_fname):
        #     print(f'Subject {subject_code} is processed')
        #     continue

        if not(subject_code in to_recompute):
            continue

        if subject_code in bad_subjects and not(subject_code in to_recompute):
            print(f'Passing subject {subject_code}')
            continue
        else:
            print(f'Processing subject {subject_code}')

        montage_filename = os.path.join(subject.dirname,  'sub-{}_montage.tcsv'.format(subject.entities['subject']))
        data_filename = subject.path
        
        subj_montage = load_montage(data_root, subject_code)
        bipolar_mask = subj_montage['name'].apply(is_bipolar)
        subj_montage = subj_montage[bipolar_mask]
        subj_chans = subj_montage['name'].tolist()
        
        subj_mop = load_mop(data_root, subject_code, 400)
        subj_mop = fix_mop(subj_mop, bipolar_mask)
        subj_parcels = subj_mop['parcel_assign']
        
        if np.array_equal(subj_chans, subj_mop['channel_names']) == False:
            index_mapper = [subj_chans.index(n) for n in subj_mop['channel_names']]
            subj_parcels = subj_parcels[index_mapper]

        bipo = make_bipolar(data_filename, montage_filename, analysis_params['lowpass_filter'])
        
        to_exclude = [ch for ch in bipo.ch_names if not(ch in subj_chans)]
        bipo.drop_channels(to_exclude)

        n_chans = len(bipo.ch_names)
        amplitude_spectrum = np.zeros((len(frequencies), n_chans), dtype=float)
        
        data_gpu = cp.array(bipo._data[:, 10000:], dtype=np.float32)

        f_psd, pxx = sp.signal.welch(bipo._data, fs=1000, nperseg=256*16)

        for freq_idx, frequency in enumerate(tqdm.tqdm(frequencies, leave=False, desc='Subject {}'.format(subject.entities['subject']))):
            data_complex = filter_data(data_gpu, 1000, frequency, omega=7.5, n_jobs='cuda')
            data_envelope = cp.abs(data_complex)

            for (s,e) in [(0, n_chans//2), (n_chans//2, n_chans)]:
                amplitude_spectrum[freq_idx, s:e] = data_envelope[s:e].mean(axis=-1).get()

        del data_gpu
        del data_envelope
        del data_complex

        cp._default_memory_pool.free_all_blocks()
            
        res = {'frequencies': frequencies, 'ch_names': bipo.ch_names,
                'amplitude_values': amplitude_spectrum, 
                'psd_frequencies': f_psd, 'psd': pxx}

        pickle.dump(res, open(result_fname, 'wb'))

if __name__ == '__main__':
    main()

