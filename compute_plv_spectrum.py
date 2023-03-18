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

def fix_mop(mop, bipolar_mask):
    res = dict()

    res['mOp'] = mop['mOp'][bipolar_mask][:, bipolar_mask]
    res['parcel_assign'] = mop['parcel_assign'][bipolar_mask]
    res['channel_names'] = [name for (name, flag) in zip(mop['channel_names'], bipolar_mask) if flag]

    return res
 
def main():
    analysis_params = json.load(open('spectrum_config.json'))

    data_root = os.path.join(analysis_params['data_path'])
    layout = BIDSLayout(data_root)

    frequencies = get_frequencies()

    bad_subjects = ['02', # weird mop
                    '06',
                    '40',
                    '42',
                    '57']

    for subject in layout.get(target='subject', extension='edf'): 
        subject_code = subject.entities['subject']
        result_fname = os.path.join(analysis_params['output_path'], 'sub-{}_spectrum.pickle'.format(subject.entities['subject'])) 
        
        if os.path.exists(result_fname):
            print(f'Subject {subject_code} is processed')
            continue

        if subject_code in bad_subjects:
            # print(f'Passing subject {subject_code}')
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

        ref_mask = create_bipolar_ref_mask(bipo.ch_names).astype(int)

        n_chans = len(bipo.ch_names)

        cplv_spectrum = np.zeros((len(frequencies), n_chans, n_chans), dtype=np.complex)
        dfa_spectrum = np.zeros((len(frequencies), n_chans), dtype=float)
        
        data_gpu = cp.array(bipo._data[:, 10000:], dtype=np.float32)

        for freq_idx, frequency in enumerate(tqdm.tqdm(frequencies, leave=False, desc='Subject {}'.format(subject.entities['subject']))):
            data_complex = filter_data(data_gpu, 1000, frequency, omega=7.5, n_jobs='cuda')
            data_envelope = cp.abs(data_complex)
            
            data_complex /= data_envelope
            
            freq_cplv = cp.inner(data_complex, cp.conj(data_complex)) / data_complex.shape[-1]

            cplv_spectrum[freq_idx] = freq_cplv.get()
                    
            samples_per_cycle = 1000/frequency
            dfa_win_lengths = np.geomspace(20*samples_per_cycle, data_gpu.shape[1]/4, 30)

            # for (s,e) in [(0, n_chans//4), (n_chans//4, n_chans//2), (n_chans//2, n_chans//4*3), (n_chans//4*3, n_chans)]:
            #     dfa_spectrum[freq_idx, s:e] = dfa_fft(data_envelope[s:e], dfa_win_lengths)[2]

        del data_gpu
        del data_envelope
        del data_complex

        cp._default_memory_pool.free_all_blocks()

        plv_morphed = morph_plv(cplv_spectrum, subj_parcels, ref_mask, n_parcels=493)
        dfa_morphed = morph_dfa(dfa_spectrum, subj_parcels, ref_mask, n_parcels=493)
            
        res = {'frequencies': frequencies, 'ch_names': bipo.ch_names,
                'plv_morphed': plv_morphed, 'dfa_morphed': dfa_morphed,
                'cplv_values': cplv_spectrum, 'dfa_values': dfa_spectrum}

        pickle.dump(res, open(result_fname, 'wb'))

if __name__ == '__main__':
    main()