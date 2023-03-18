import numpy as np
import pandas as pd

import pickle
import os

# def load_mop(data_root, subj_num, n_cortical=100):
#     subj_name = 'sub-{}'.format(subj_num)
    
#     fname = os.path.join(data_root, subj_name, 'ses-01', 'ieeg', f'{subj_name}_par-Schaefer2018_{n_cortical}Parcels_17Networks_order_mOp.npy')
#     return np.load(fname, allow_pickle=True)

def load_mop(data_root, subj_num, n_cortical=100):
    subj_name = 'sub-{}'.format(subj_num)
    
    fname = os.path.join(data_root, subj_name, 'ses-01', 'ieeg', f'sub-{subj_num}_par-parc2018yeo7_{n_cortical}_mOp.npy')
    
    return np.load(fname, allow_pickle=True)

def load_yeo(data_root, subj_num):
    subj_name = 'sub-{}'.format(subj_num)
    fname = os.path.join(data_root, subj_name, 'ses-01', 'ieeg', f'{subj_name}_par-yeo7_mOp.npy')
    return np.load(fname, allow_pickle=True)['mOp']


def load_electrodes(data_root, subj_num):
    subj_name = 'sub-{}'.format(subj_num)
    fname = os.path.join(data_root, subj_name, 'ses-01', 'ieeg', f'{subj_name}_electrodes.tcsv')
    return pd.read_csv(fname, sep='\t').set_index('name')


def load_montage(data_root, subj_num):
    subj_name = 'sub-{}'.format(subj_num)
    fname = os.path.join(data_root, subj_name, 'ses-01', 'ieeg', f'{subj_name}_montage.tcsv')
    return pd.read_csv(fname, sep='\t')


def load_psd_data(subject_number):
    return pickle.load(open(f'meso_data/psd_statistics/sub-{subject_number}_spectrum.pickle', 'rb'))


def is_bipolar(ch_name):
    a,c = ch_name.split('-')
    return len(a) > 0 and len(c) > 0


def fix_mop(mop, bipolar_mask):
    res = dict()

    res['mOp'] = mop['mOp'][bipolar_mask][:, bipolar_mask]
    res['parcel_assign'] = mop['parcel_assign'][bipolar_mask]
    res['channel_names'] = [name for (name, flag) in zip(mop['channel_names'], bipolar_mask) if flag]

    return res

def read_seeg_parcel_names(n_cortical, root='meso_data'):
    fpath = os.path.join(root, f'Schaefer2018_{n_cortical}Parcels_17Networks_order_LUT.txt')
    parcel_names = list()

    with open(fpath) as fin:
        for line in fin.readlines():
            tokens = line.strip().split()
            name = tokens[1]
            parcel_names.append(name)
    
    return parcel_names
            