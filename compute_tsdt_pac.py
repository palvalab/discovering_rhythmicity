import os
import pickle
import logging

from itertools import count

import numpy as np
import pandas as pd
import cupy as cp

import tqdm.notebook as tqdm
import mne

### UGLY ### 
import sys
import os.path as op
source_dir = op.join('L:/', 'nttk-data3', 'palva', 'Common repos', 'OL2015','source',
                    'Python37','Utilities')
sys.path.append(source_dir)

import preprocessor.filehandling as fh
###

from utils.tsdt import EpochPACFConvolveBootstrap

def get_cue_meta(subject_name, basename, hands_df):
    tokens = basename.split('_')
    cue = tokens[4]
    para = tokens[3]

    tsdt_type = basename[20:28]
    hand = hands_df.loc[subject_name, tsdt_type]
    
    return cue, para, hand

def main():
    cp.cuda.Device(1).use()

    sfreq = 1000
    lags_cycles_similarity = np.arange(1, 2, 0.05)
    window_size = 2.5

    frequencies = np.geomspace(5, 100, 80)

    parcellation = 'parc2018yeo17_200'

    subjects_dir = os.path.join('L:/','nttk_tsdt_tdt','TSDT_TDT_BIDS','derivatives','')
    mne.utils.set_config("SUBJECTS_DIR", subjects_dir, set_env=True)


    logging_dir = os.path.join('L:/','nttk_tsdt_tdt','TSDT_TDT_BIDS','derivatives','_logging')

    log_dir = os.path.join(logging_dir, 'epochs.log')
    logging.basicConfig(format='%(asctime)s %(message)s', filename=log_dir, level=logging.ERROR)


    hands = pd.read_csv(os.path.join(subjects_dir, '_documentation', 'hands.csv'), sep=';',
                                    index_col= ['Subject'])


    subjects = fh.get_subjects(subjects_dir, excl_strings=['_'])

    subjects_filtered = [s for s in subjects if os.path.exists(os.path.join(subjects_dir, s, 'parcel_ts2'))]

    output_path = os.path.join('L:', 'nttk-data3', 'palva', 'Vlad', 'pac_meg_tsdt_tfr_5.0_convolve_boot_balanced')
    bar = tqdm.tqdm(subjects_filtered)
    for subject in bar:   
        
        # Define all the correct sequences here
        cor_seqs = {'att': {'hit':          [(4,), (1,2), (8,)], # NO RESPONSE YET, Comma important
                            'miss':         [(4,), (1,2), (8,)]},
                    'res': {'hit':                [(1,2), (8,)],
                            'miss':               [(1,2), (8,)]},
                    'non': {'hit':                [(1,2)],
                            'miss':               [(1,2)]}}

        codes = {'att': {'hit':           {'stim': (1,2)},
                        'miss':           {'stim': (1,2)}},
                'res': {'hit':            {'stim': (1,2)},
                        'miss':           {'stim': (1,2)}},
                'non': {'hit':            {'stim': (1,2)},
                        'miss':           {'stim': (1,2)}}}

        times = {'att': {'hit':          {'stim': (-1500, 1500)},
                        'miss':          {'stim': (-1500, 1500)}},
                'res': {'hit':           {'stim': (-1500, 1500)},
                        'miss':          {'stim': (-1500, 1500)}},
                'non': {'hit':           {'stim': (-1500, 1500)},
                        'miss':          {'stim': (-1500, 1500)}}}


        length = {  'att': 3000,
                    'res': 3000,
                    'non': 3000}



        pts_list, pts_list2 = fh.get_files(subjects_dir, [subject], 'parcel_ts2', 
                                        incl_strings=['TSDT', 'ica-py'],
                                        excl_strings=[], ftype='.npy')

        event_list, event_list2 = fh.get_files(subjects_dir, [subject], 'events2', 
                                        incl_strings=['TSDT', 'ica-py'], excl_strings=[],
                                        ftype='.npy')

        total = len([x for i, x in enumerate(pts_list2) if x[11]!='b'])
        t2 = tqdm.tqdm(zip(count(), pts_list, event_list), total=total, leave=False, disable=True)
        for i, pts_file, event_file in t2:
            pts1 = np.load(pts_file)
            events1 = np.load(event_file)
            
            labels = mne.read_labels_from_annot(subject, subjects_dir=subjects_dir, parc=parcellation, verbose=False)

            # CHECK FOR PARTS
            if pts_list2[i][11] == 'a':
                pts2 = np.load(pts_list[i+1])
                pts_list.remove(pts_list[i+1])
                pts_list2.remove(pts_list2[i+1])

                events2 = np.load(event_list[i+1])
                event_list.remove(event_list[i+1])
                event_list2.remove(event_list2[i+1])

            else:
                pts2 = None
                events2 = None

            # MAKE THE EPOCH CLASS
            epoch = EpochPACFConvolveBootstrap(sfreq, None, pts_list2[i][:-4],
                    [pts1, pts2], [events1, events2])

            bar.set_description(f'Processing subject {subject}, set {epoch.set_name}')


            subject_output = os.path.join(output_path, subject)

            if not(os.path.exists(subject_output)):
                os.mkdir(subject_output)
                
            subject_output = os.path.join(output_path, subject)
            set_output_name = os.path.join(subject_output, epoch.set_name + '.npy')

            if os.path.exists(set_output_name):
                print(f'Subject {subject}, Set {epoch.set_name} is processed already!')
                continue

            # HANDS AND CORRECT ANSWERS
            key = pts_list2[i][25:28]
            hand = hands.loc[subject, pts_list2[i][20:28]]

            if hand == 'left':
                cor_seqs[key]['hit'].append((16,))

                if key == 'non':
                    cor_seqs[key]['miss'].append((1,2))

                else:
                    cor_seqs[key]['miss'].append((64,))


            elif hand == 'right':
                cor_seqs[key]['hit'].append((32,))

                if key == 'non':
                    cor_seqs[key]['miss'].append((1,2))

                else:
                    cor_seqs[key]['miss'].append((128,))


            else:
                print("Hand not specified for {}".format(pts_list2[i]))
                logging.error("Hand not specified for {}".format(subject))
                break


            # PARSE SEQUENCES
            epoch.parse_sequences(cor_seqs[key])

            # INTERVALs
            epoch.make_intervals(times[key], codes[key], length[key])
            
            epoch.make_epochs(frequencies, 
                            lags_cycles_similarity, window_size=window_size, disable_tqdm=True,
                            epoch_types=['hit','miss'], n_cycles=5)
            # epoch.make_epochs(frequencies, max_lag, ['hit','miss'], n_cycles=5)

            epoch.epochs['cue'], epoch.epochs['para'], epoch.epochs['hand'] = get_cue_meta(subject, epoch.set_name, hands)
            epoch.epochs['set_name'] = epoch.set_name
            epoch.epochs['parcel_names'] = [l.name for l in labels if '17Networks' in l.name]
            epoch.epochs['frequencies'] = frequencies
            epoch.epochs['prestim'] = -1500
            epoch.epochs['poststim'] = 1500

            pickle.dump(epoch.epochs, open(set_output_name, 'wb'))

        bar.update(1)


if __name__ == '__main__':
    main()