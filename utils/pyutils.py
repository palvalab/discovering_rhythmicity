import os
import subprocess

import numpy as np

from collections import defaultdict


def group_list_indices(values, second_grouper):
    res = defaultdict(lambda: defaultdict(list))
    
    for idx, (v, g) in enumerate(zip(values, second_grouper)):
        res[v][g].append(idx)
    
    # remove unknown parcel
    if -1 in res:
        res.pop(-1)
    
    return res

def random_choice_list(arr, size):
    indices = np.arange(len(arr))
    choice_idx = np.random.choice(indices, size=size)
    
    return [arr[i] for i in choice_idx]

def group_list_indices_by_subjects(values, second_grouper):
    res = defaultdict(lambda: defaultdict(list))
    
        
    for subj_idx, (subj_values, subj_grouper) in enumerate(zip(values, second_grouper)):
        for contact_idx, (v,g) in enumerate(zip(subj_values, subj_grouper)):
            res[v][g].append((subj_idx, contact_idx))
    
    # remove unknown parcel
    if -1 in res:
        res.pop(-1)
    
    return res


def reorder(arr, labels_src, labels_dst, axis=0):
    indices = [labels_src.index(n) for n in labels_dst]
    return np.take(arr, indices, axis=axis)

def convert_to_array(objects_list):
    res = np.empty(len(objects_list), dtype=object)
    res[:] = objects_list
    
    return res

def nanop(arr, op, value):
    if np.isnan(arr):
        return np.nan
    
    res = eval(f'arr {op} value')
    return res


def get_subsystem(name):
    return name.split('_')[2]

def get_subsystem_hemi(name):
    return '_'.join(name.split('_')[1:3])

def create_subsystem_map(values):
    res = dict()
    
    for v in values:
        if not(v in res):
            res[v] = len(res)
    
    return res

import subprocess

def call_with_output(command):
    success = False
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode()
        success = True 
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
    except Exception as e:
        # check_call can raise other exceptions, such as FileNotFoundError
        output = str(e)
    return(success, output)


def plot_as_emf(figure, fpath, **kwargs):
    inkscape_path = 'D:\\Vlad\\projects\\inkscape\\bin\\inkscape.com'

    path, filename = os.path.split(fpath)
    filename, extension = os.path.splitext(filename)

    svg_filepath = os.path.join(path, filename+'.svg')
    emf_filepath = os.path.join(path, filename+'.emf')

    figure.savefig(svg_filepath, format='svg')

    out = call_with_output([inkscape_path, '--export-type=wmf', svg_filepath])
