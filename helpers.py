#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:06:41 2018

@author: bitzer
"""

import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datadir = "../data/behaviour/Data/raw"
resultsdir = "../inf_results/behaviour"

dt = 0.01
maxrt = 2.0
choices = dict(left=-1, right=1)
toresponse = [0, 3.0]

def find_available_subjects(datadir=datadir):
    """checks directories for data files (one per subject) and returns 
    available subjects."""
    
    _, _, filenames = next(os.walk(datadir))
    subjects = []
    for fname in filenames:
        match = re.match('^s(\d+)_main_data.txt$', fname)
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
    
    
def diff_diff(ch, rt, easy, correct):
    """Computes the difference in accuracy and median RT between two 
       difficulty levels."""
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