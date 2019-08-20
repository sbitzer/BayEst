#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:14:49 2019

@author: bitzer
"""
from __future__ import print_function

import numpy as np
import pandas as pd
import os
from glob import glob

import helpers
import snl_simulators as snlsim

import matplotlib.pyplot as plt
import seaborn as sns


#%% selection of results
stats = 'hist'

# base model
results = {
        'ED': os.path.join(
                helpers.resultsdir, 'snl/rotated_directions/201807061824'),
        'EEG': os.path.join(
                helpers.resultsdir, 'snl/rotated_directions/201907240843'),
        'HDDM': os.path.join(
                helpers.resultsdir, 'hddm', 'hddm_result_201807101804.h5')}

subjects = [int(os.path.basename(f)[1:3]) 
            for f in glob(os.path.join(results['EEG'], '*%s.log' % stats))]
subjects = sorted(subjects)[:-1]


#%% estimate goodness of fit (samples from posterior)
#within_deviance = lambda x: np.sum(x**2, axis=1)
measures = pd.Index(['D', 'MSE'], name='measure')
models = pd.Index(['HDDM', 'ED', 'EEG'], name='model')

fit = pd.DataFrame(
        index=pd.Index(subjects, name='subject'), 
        columns=pd.MultiIndex.from_product(
                [measures, models], names=['measure', 'model']))

across_deviance = lambda x: np.mean(np.abs(x), axis=0)

for measure in measures:
    if measure == 'D':
        within_deviance = lambda x: np.median(np.abs(x), axis=1)
    elif measure == 'MSE':
        within_deviance = lambda x: np.sum(x ** 2, axis=1)
    
    print('computing %s for:' % measure)

    for model in models:
        print('    ' + model + ' ...')
        fit.loc[:, (measure, model)], eh = snlsim.estimate_posterior_fit(
                results[model], subjects, stats, within_deviance,
                across_deviance)

with pd.HDFStore('figures/goodness_of_fit.h5', 'w') as store:
    store['results'] = pd.Series(results)
    store['fit'] = fit
    

#%% plot goodness of fit values across subjects
#fig, ax = plt.subplots(figsize=(6, 4))
#sns.distplot(fit_HDDM, rug=True, ax=ax, label='HDDM')
#
#sns.distplot(fit_ED, rug=True, ax=ax, label='ED')
#sns.distplot(fit_EEG, rug=True, ax=ax, label='EEG')
#
#ax.legend()
#ax.set_xlabel(measure + ' across subjects')
#ax.set_ylabel('density estimate')
#
#fig.tight_layout()
#fig.savefig('figures/posterior_fit_' + measure + '_ED_EEG_HDDM.png', dpi=300)
#
#
##%% plot how well the fit captures the difference between easy and hard 
##   conditions across subjects
#fig_eh, ax = plt.subplots()
#
#sns.distplot([eh_HDDM.map(lambda lx: lx[2])], rug=True, ax=ax, label='HDDM')
#
#sns.distplot([eh_ED.map(lambda lx: lx[2])], rug=True, ax=ax, label='ED')
#sns.distplot([eh_EEG.map(lambda lx: lx[2])], rug=True, ax=ax, label='EEG')
#
#ax.legend()
#ax.set_xlabel('deviance of easy-hard difference')