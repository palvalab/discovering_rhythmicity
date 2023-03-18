from collections import defaultdict

import numpy as np
import cupy as cp

import tqdm

### UGLY ####
import sys
import os.path as op
source_dir = op.join('L:/', 'nttk-data3', 'palva', 'Common repos', 'OL2015','source',
                    'Python37','Utilities')
sys.path.append(source_dir)

from preprocessor.epoching import Epoch
#####

from crosspy.preprocessing.signal import filter_data
from crosspy.core.phase import compute_instantaneous_frequency

from utils.pac import _pac_kernel
from utils.stats import moving_average_fast

class EpochPACF(Epoch):    
    def make_epochs(self, frequencies, lags_cycles,
                    epoch_types, 
                    n_cycles=4.5, n_jobs='cuda', disable_tqdm=True):
        
        n_freqs = len(frequencies)
        
        self.epochs = defaultdict(dict)
        stat_types = ['pacf', 'amplitude', 'evoked']
        
        for epoch_type in epoch_types:
            for stat_type, dtype in zip(stat_types, [float, float, np.complex64]):
                self.epochs[epoch_type][stat_type] = np.zeros((n_freqs, self.pts.shape[0], self.intervals[epoch_type].shape[-1]), dtype=dtype)
        
        for start, end in zip([0, 100], [100, 200]):
            data = cp.array(self.pts[start:end], dtype=np.float32)

            phase_diff = cp.zeros(data.shape, dtype=cp.complex64)

            for freq_idx, freq in enumerate(tqdm.tqdm(frequencies, leave=False, disable=disable_tqdm)):
                data_complex = filter_data(data, self.sfreq, freq, n_cycles, n_jobs)
                
                data_envelope = cp.abs(data_complex)
                data_normed = data_complex / data_envelope
                data_conj = cp.conj(data_normed)

                phase_similarity = cp.zeros_like(phase_diff)

                data_if = compute_instantaneous_frequency(data_normed, self.sfreq).mean(axis=-1).get()

                lags_samples = cp.array((lags_cycles.reshape(-1, 1) * (self.sfreq / data_if).reshape(1,-1)).astype(int))

                for lag_idx, lags_chanwise in enumerate(lags_samples):
                    _ = _pac_kernel(data_normed, data_conj, lags_chanwise, data_complex.shape[-1], phase_diff)
                    phase_similarity += phase_diff

                phase_similarity /= len(lags_cycles)
                phase_similarity = cp.abs(phase_similarity)

                for epoch_type in epoch_types:
                    for stat_type, stat_array in zip(stat_types, [phase_similarity, data_envelope, data_normed]):
                        vals = stat_array[..., self.intervals[epoch_type]].mean(axis=1).get()
                        self.epochs[epoch_type][stat_type][freq_idx, start:end] = vals

            if n_jobs == 'cuda':
                del data
                del phase_diff
                del data_complex
                del data_envelope
                del data_normed 
                del data_conj
                del phase_similarity
                del data_if

                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()

class EpochPACFConvolve(Epoch):    
    def make_epochs(self, frequencies, 
                    lags_cycles, window_size,
                    epoch_types, 
                    n_cycles=4.5, n_jobs='cuda', disable_tqdm=False):
        
        n_freqs = len(frequencies)
        n_lags = len(lags_cycles)

        self.epochs = defaultdict(dict)
        stat_types = ['pacf', 'amplitude', 'evoked']
        
        for epoch_type in epoch_types:
            for stat_type, dtype in zip(stat_types, [float, float, np.complex64]):
                self.epochs[epoch_type][stat_type] = np.zeros((n_freqs, self.pts.shape[0], self.intervals[epoch_type].shape[-1]), dtype=dtype)
        
        for start, end in zip([0, 100], [100, 200]):
            data = cp.array(self.pts[start:end], dtype=np.float32)

            phase_diff = cp.zeros(data.shape, dtype=cp.complex64)

            for freq_idx, freq in enumerate(tqdm.tqdm(frequencies, leave=False, disable=disable_tqdm)):
                data_complex = filter_data(data, self.sfreq, freq, n_cycles, n_jobs)
                
                data_envelope = cp.abs(data_complex)
                data_normed = data_complex / data_envelope
                data_conj = cp.conj(data_normed)
                
                phase_similarity = cp.zeros_like(phase_diff, dtype=float)

                data_if = compute_instantaneous_frequency(data_normed, self.sfreq).mean(axis=-1)

                for lag_idx, lag in enumerate(lags_cycles):
                    lag_samples = cp.rint(lag*self.sfreq/data_if).astype(int)

                    _pac_kernel(data_normed, data_conj, lag_samples, data.shape[-1], phase_diff)
                    phase_diff[cp.isnan(phase_diff)] = 0.0

                    window_size_samples = int(cp.rint(window_size*self.sfreq/data_if).mean())
                    phase_similarity += cp.abs(moving_average_fast(phase_diff, window_size_samples))
                
                phase_similarity /= n_lags

                for epoch_type in epoch_types:
                    for stat_type, stat_array in zip(stat_types, [phase_similarity, data_envelope, data_normed]):
                        vals = stat_array[..., self.intervals[epoch_type]].mean(axis=1).get()
                        self.epochs[epoch_type][stat_type][freq_idx, start:end] = vals


            if n_jobs == 'cuda':
                del data
                del phase_diff
                del data_complex
                del data_envelope
                del data_normed 
                del data_conj
                del phase_similarity
                del data_if

                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()


class EpochPACFConvolveBootstrap(Epoch):    
    def make_epochs(self, frequencies, 
                    lags_cycles, window_size,
                    epoch_types, 
                    n_cycles=4.5, n_jobs='cuda', disable_tqdm=False):
        
        n_freqs = len(frequencies)
        n_lags = len(lags_cycles)

        self.epochs = defaultdict(dict)
        stat_types = ['pacf', 'amplitude', 'evoked']

        n_min_trials = min([self.intervals[epoch_type].shape[0] for epoch_type in epoch_types])
        
        for epoch_type in epoch_types:
            for stat_type, dtype in zip(stat_types, [float, float, np.complex64]):
                self.epochs[epoch_type][stat_type] = np.zeros((n_freqs, self.pts.shape[0], self.intervals[epoch_type].shape[-1]), dtype=dtype)
        
        for start, end in zip([0, 100], [100, 200]):
            data = cp.array(self.pts[start:end], dtype=np.float32)

            phase_diff = cp.zeros(data.shape, dtype=cp.complex64)

            for freq_idx, freq in enumerate(tqdm.tqdm(frequencies, leave=False, disable=disable_tqdm)):
                data_complex = filter_data(data, self.sfreq, freq, n_cycles, n_jobs)
                
                data_envelope = cp.abs(data_complex)
                data_normed = data_complex / data_envelope
                data_conj = cp.conj(data_normed)
                
                phase_similarity = cp.zeros_like(phase_diff, dtype=float)

                data_if = compute_instantaneous_frequency(data_normed, self.sfreq).mean(axis=-1)

                for lag_idx, lag in enumerate(lags_cycles):
                    lag_samples = cp.rint(lag*self.sfreq/data_if).astype(int)

                    _pac_kernel(data_normed, data_conj, lag_samples, data.shape[-1], phase_diff)
                    phase_diff[cp.isnan(phase_diff)] = 0.0

                    window_size_samples = int(cp.rint(window_size*self.sfreq/data_if).mean())
                    phase_similarity += cp.abs(moving_average_fast(phase_diff, window_size_samples))
                
                phase_similarity /= n_lags

                for epoch_type in epoch_types:
                    type_intervals = self.intervals[epoch_type]
                    trial_size = self.intervals[epoch_type].shape[-1]

                    for stat_type, stat_array in zip(stat_types, [phase_similarity, data_envelope, data_normed]):
                        
                        if type_intervals.shape[0] == n_min_trials:
                            vals = np.abs(stat_array[..., type_intervals].mean(axis=1).get())
                        else:
                            vals = self._bootstrap_statistic(stat_array, self.intervals[epoch_type], n_min_trials, n_freqs, trial_size, n_rounds=100)

                        self.epochs[epoch_type][stat_type][freq_idx, start:end] = vals


            if n_jobs == 'cuda':
                del data
                del phase_diff
                del data_complex
                del data_envelope
                del data_normed 
                del data_conj
                del phase_similarity
                del data_if

                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()

    def _bootstrap_statistic(self, stat_array, type_intervals, n_min_trials, n_freqs, trial_size, n_rounds=100):
        res = cp.zeros((stat_array.shape[0], trial_size))

        indices = np.arange(type_intervals.shape[0])

        for round_idx in range(n_rounds):
            round_indices = np.random.choice(indices, size=n_min_trials, replace=False)
            round_intervals = type_intervals[round_indices]

            res += cp.abs( stat_array[..., round_intervals].mean(axis=1) )
        
        res /= n_rounds

        return res.get()