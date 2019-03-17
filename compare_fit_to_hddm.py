#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:14:49 2019

@author: bitzer
"""

import numpy as np
import os
from glob import glob

import helpers
import snl_simulators as snlsim

import matplotlib.pyplot as plt
import seaborn as sns


#%% selection of results
stats = 'hist'

# base model
result = os.path.join(
        helpers.resultsdir, 'snl/rotated_directions/201807061824')

subjects = [int(os.path.basename(f)[1:3]) 
            for f in glob(os.path.join(result, '*%s.log' % stats))]
subjects = sorted(subjects)[:-1]

hddm_result = os.path.join(
        helpers.resultsdir, 'hddm', 'hddm_result_201807101804.h5')


#%% estimate goodness of fit (samples from posterior)
#within_deviance = lambda x: np.sum(x**2, axis=1)
within_deviance = lambda x: np.median(np.abs(x), axis=1)

across_deviance = lambda x: np.mean(np.abs(x), axis=0)

fit, eh = snlsim.estimate_posterior_fit(
        hddm_result, subjects, stats, within_deviance, across_deviance)

fit_comp, eh_comp = snlsim.estimate_posterior_fit(
        result, subjects, stats, within_deviance, across_deviance)


#%% plot goodness of fit values across subjects
fig, ax = plt.subplots()
sns.distplot(fit, rug=True, ax=ax, label='HDDM')

sns.distplot(fit_comp, rug=True, ax=ax, label='our model')

ax.legend()
ax.set_xlabel('estimated deviance from data')


#%% plot how well the fit captures the difference between easy and hard 
#   conditions across subjects
fig_eh, ax = plt.subplots()

sns.distplot([eh.map(lambda lx: lx[2])], rug=True, ax=ax, label='HDDM')

sns.distplot([eh_comp.map(lambda lx: lx[2])], rug=True, ax=ax, label='our model')

ax.legend()
ax.set_xlabel('deviance of easy-hard difference')