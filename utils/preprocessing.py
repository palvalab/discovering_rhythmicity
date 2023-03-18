import re
import os

import cupy as cp
import numpy as np
import scipy as sp
import pandas as pd

import itertools

from cupy.cuda import cufft

try:
    import cusignal
except:
    print('cusignal not found!')

import mne

from crosspy.preprocessing.seeg.support import clean_montage, drop_monopolar_channels
from crosspy.preprocessing.seeg.seeg_utils import create_reference_mask

def _centered(arr: cp.ndarray, newshape):
    # Return the center newshape portion of the array.
    newshape = cp.asarray(newshape)
    currshape = cp.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _fftconvolve_simple(in1: cp.ndarray, in2: cp.ndarray):
    shape = np.array(in1.shape) + np.array(in2.shape) - 1
    shape_fast = tuple(sp.fftpack.next_fast_len(s) for s in shape)
    fslice = tuple([slice(sz) for sz in shape])
    
    r1 = cp.fft.fftn(in1, shape_fast, axes=[0])
    r2 = cp.fft.fftn(in2, shape_fast, axes=[0])

    res = cp.fft.ifftn(r1*r2)[fslice]
    return _centered(res, in1.shape)

def routine_gpu(data_gpu, sr, omega, morlet_frequency, data_buffer=None):
    n_chans, n_ts = data_gpu.shape

    win = cp.array(mne.time_frequency.morlet(sr, [morlet_frequency], omega)[0])
    win /= (np.abs(win).sum()/2)
    
    if data_buffer is None:
        data_preprocessed = cp.zeros_like(data_gpu, dtype=cp.complex64)
    else:
        data_preprocessed = data_buffer
    
    for i in range(n_chans):
        data_preprocessed[i] = cusignal.fftconvolve(data_gpu[i], win, 'same')
            
    return data_preprocessed


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

    return bipo


def load_bipolar(subject):
    montage_filename = os.path.join(subject.dirname,  'sub-{}_montage.tcsv'.format(subject.entities['subject']))
    electrodes_filename = os.path.join(subject.dirname, 'sub-{}_electrodes.tcsv'.format(subject.entities['subject']))
    meta_filename =  os.path.join(subject.dirname, 'sub-{}_meta.csv'.format(subject.entities['subject']))
    data_filename = subject.path
    
    data_ref =  make_bipolar(data_filename, montage_filename)
    
    notch_freqs = np.arange(50, data_ref.info['sfreq']//2, 50)

    # data_ref.notch_filter(notch_freqs, verbose=False)
    
    return data_ref


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