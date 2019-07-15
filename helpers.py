#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:06:41 2018

@author: bitzer
"""

import os, re
from glob import glob
import pandas as pd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from rotated_directions import identify_model

eegdir = "../data/EEG"
datadir = "../data/behaviour/Data/raw"
resultsdir = "../inf_results/behaviour"

dt = 0.01
maxrt = 2.0
choices = dict(left=-1, right=1)
toresponse = [0, 3.0]

def find_available_subjects(datadir=datadir, eegdir=None):
    """checks directories for data files (one per subject) and returns 
    available subjects. When `eegdir`is given only subjects with available
    EEG data will be returned."""
    
    if eegdir is not None:
        datadir = eegdir
    
    _, _, filenames = next(os.walk(datadir))
    subjects = []
    for fname in filenames:
        if eegdir is None:
            match = re.match('^s(\d+)_main_data.txt$', fname)
        else:
            match = re.match('^s(\d+)_chresp_Unsrtd.h5$', fname)
        if match is not None:
            subjects.append(int(match.group(1)))
    
    return np.sort(subjects)


def load_subject(sub, exclude_to=False, censor_late=True):
    file = os.path.join(datadir, 's%02d_main_data.txt' % sub)
    
    data = pd.read_csv(file, '\t')
    
    assert np.all(data.dotCoh == data.dotCoh[0]), ("subject has differing dot "
                                                   "coherences!")
    
    data.response = data.response.map(choices, 'ignore')
    
    timeouts = data.RT.isna()
    data.loc[timeouts, 'RT'] = toresponse[1]
    data.loc[timeouts, 'response'] = toresponse[0]
    
    data['easy'] = np.abs(data.tarDir - data.critDir) > 20
    
    # ensure that directions are in valid ranges and adhere to model definition
    data.tarDir = -data.tarDir % 360
    data.critDir = -data.critDir % 180
    
    if censor_late:
        timeouts = data.RT > maxrt
        data.loc[timeouts, 'RT'] = toresponse[1]
        data.loc[timeouts, 'response'] = toresponse[0]
        data.loc[timeouts, 'error'] = np.nan
        
    if exclude_to:
        data.dropna(inplace=True)
    
    # copy so that not the full dataframe is kept in memory, because the 
    # returned object is otherwise only a reference to the full dataframe
    return data[['easy', 'tarDir', 'critDir', 'RT', 'response', 'error']
                ].copy()
    
    
def load_subject_eeg(sub, olddt=1/256.):
    """Loads EEG data, resamples to set dt, brings into `Trials` format
    and scales to std=1 across all data."""
    fpath = os.path.join(eegdir, 's%02d_chresp_Unsrtd.h5' % sub)
    
    with pd.HDFStore(fpath, 'r') as store:
        data = store.root.CHRSP.read()
        trind = store.root.TRLS.read()
        channels = store.root.CHNLS.read()
        
    oldt = np.arange(0, data.shape[2] * olddt, olddt)
    
    Trials, newt = scipy.signal.resample(
            data.transpose(2, 1, 0), int(round(oldt[-1] / dt)), oldt)
            
    return Trials / Trials.std(), trind, channels
    
    
def diff_diff(ch, rt, easy, correct):
    """Computes the difference in accuracy and median RT between two 
       difficulty levels."""
    ch = np.array(ch)
    rt = np.array(rt)
       
    if ch.ndim == 1 and rt.ndim == 1:
        ch = ch[:, None]
        rt = rt[:, None]
        
    if correct.ndim == 1:
        correct = correct[:, None]
    
    ch_corr = ch == correct
    
    return (ch_corr[easy, :].mean(axis=0) - ch_corr[~easy, :].mean(axis=0), 
            np.median(rt[easy, :], axis=0) - np.median(rt[~easy, :], axis=0))
    
    
def plot_diffs(choices, rts, condind, correct, data, label='', in_diffs=False, 
               ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        
    match = re.match(r'diffbox_(\d+)', ax.aname)
    if match:
        add = int(match.group(1))
        ax.aname = 'diffbox_{}'.format(add+1)
    else:
        ax.aname = 'diffbox_1'
        add = 0
        
    if in_diffs:
        diffacc = choices
        diffmed = rts
    else:
        diffacc, diffmed = diff_diff(
                choices, rts, condind, correct)
    
    positions = np.r_[1, 2] + 0.2 * add
        
    ax.boxplot(np.c_[diffacc, diffmed], positions=positions)
    
    diffacc, diffmed = diff_diff(
            data.response, data.RT, condind, correct)
    
    ax.plot(np.r_[1, 2], np.r_[diffacc[0], diffmed[0]], '*', ms=10, 
            color='C0')
    
    if add:
        ax.set_xticks([1, 2])
    else:
        ax.set_ylabel('difference ({})'.format(label))
    
    ax.set_xticklabels(['accuracy', 'median RT'])
    
    return ax


def print_result_info(resultnum, stats='hist'):
    result = os.path.join(resultsdir, 'snl', 'rotated_directions', resultnum)
    
    if not os.path.isdir(result):
        raise ValueError('Could not find the given result!')
    
    subjects = sorted([int(os.path.basename(f)[1:3]) 
            for f in glob(os.path.join(result, '*%s.log' % stats))])
    
    info  = 'Result: {}\n'.format(resultnum)
    info += '=' * (len(info) - 1) + '\n\n'
    
    info += 'subjects\n'
    info += '--------\n'
    info += '{}\n\n'.format(subjects.__str__())
    
    subject = subjects[0]
    
    with pd.HDFStore(os.path.join(
            result, 's{:02d}_{}.h5'.format(subject, stats)), 'r') as store:
        info += 'data options\n'
        info += '------------\n'
        info += store.data_info.iloc[1:].__str__()[:-13]
        info += '\n'
        
        info += 'model\n'
        info += '-----\n'
        info += identify_model(list(store.parameters.columns) 
                               + list(store.fix.index)) + '\n\n'
        
        info += 'ndtdist = {}\n\n'.format(store.ndtdist[0])
        
        info += 'fix\n'
        info += '---\n'
        info += store.fix.__str__()[:-14]
        info += '\n'
        
        info += 'priors\n'
        info += '------\n'
        info += pd.concat(
                [store.prior_mu, pd.Series(
                        np.sqrt(np.diag(store.prior_cov)), 
                        index=store.prior_mu.index)], 
                axis=1).__str__()[54:]
    
    print(info)