#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
checks whether the EEG data contains information consistent with the experiment
by plotting the distribution of channel responses (EEG representation of
motion directions) relative to the true motion direction shown in the trial

a histogram centred on 0 shows that the EEG most strongly represents the true
motion direction and asserts the alignment of the motion directions defined
for EEG and behavioural data

Created on Sun Jun 16 19:20:57 2019

@author: bitzer
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import helpers

sub = 18

T, trind, channels = helpers.load_subject_eeg(sub)
data = helpers.load_subject(sub)

channels = np.round((-channels * 180 / np.pi) % 360)

Tmeans = pd.DataFrame(
        T.mean(axis=0), 
        index=channels, 
        columns=trind)

# align means on correct direction (channel)
allchs = np.unique(
        (channels[:, None] - channels[None, :]) % 360)

Tmeans_aligned = pd.DataFrame(columns=trind, index=allchs, dtype=float)

for tr in trind:
    chs = (channels - data.loc[tr, 'tarDir']) % 360
    Tmeans_aligned.loc[chs, tr] = Tmeans.loc[:, tr].values


# 
def polarbar(data):
    fig = plt.figure()
    ax = plt.subplot(111, polar=True)
    
    angles = data.index * np.pi / 180
    
    bottom = data.max()
    
    width = (2*np.pi) / data.size
    
    bars = ax.bar(angles, data, width=width, bottom=bottom)
    
    # Use custom colors and opacity
#    for r, bar in zip(radii, bars):
#        bar.set_facecolor(plt.cm.jet(r / 10.))
#        bar.set_alpha(0.8)
    
    return fig, ax


polarbar(Tmeans_aligned.mean(axis=1))