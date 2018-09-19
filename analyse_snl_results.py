#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:36:18 2018

@author: bitzer
"""

import numpy as np
import pandas as pd
import os
from glob import glob

import helpers
import snl_simulators as snlsim

import pyEPABC.parameters as parameters
from pyEPABC.parameters import exponential, gaussprob, zero

import matplotlib.pyplot as plt


#%% 
stats = 'hist'
result = os.path.join(helpers.resultsdir, 
                      'snl/rotated_directions/201809141814')
result = os.path.join(helpers.resultsdir, 
                      'snl/rotated_directions/201809172008')

subjects = [int(os.path.basename(f)[1:3]) 
            for f in glob(os.path.join(result, '*%s.log' % stats))]


#%%
with pd.HDFStore(os.path.join(
        result, 's%02d_%s.h5' % (subjects[0], stats)), 'r') as store:
    fix = store['fix']
    psamples = store['parameters']
    R = psamples.index.get_level_values('round').unique().size
    S = psamples.loc[R].shape[0]
    
pars = parameters.parameter_container()
if 'diffstd' not in fix.keys():
    pars.add_param('diffstd', 0, 10, exponential())
if 'cpsqrtkappa' not in fix.keys():
    pars.add_param('cpsqrtkappa', 0, 10, zero())
if 'critstd' not in fix.keys():
    pars.add_param('critstd', 0, 1, exponential())
if 'cnoisestd' not in fix.keys():
    pars.add_param('cnoisestd', 0, 1, exponential())
if 'dnoisestd' not in fix.keys():
    pars.add_param('dnoisestd', 0, 1.2, exponential())
if 'dirstd' not in fix.keys():
    pars.add_param('dirstd', 0, 1.2, exponential())
if 'bound' not in fix.keys():
    pars.add_param('bound', 0, 1, gaussprob(width=0.5, shift=0.5))
if 'bias' not in fix.keys():
    pars.add_param('bias', 0, .2, gaussprob())
if 'ndtloc' not in fix.keys():
    pars.add_param('ndtloc', -2, 1)
if 'ndtspread' not in fix.keys():
    pars.add_param('ndtspread', np.log(0.2), 1, exponential())
if 'lapseprob' not in fix.keys():
    pars.add_param('lapseprob', -1.65, 1, gaussprob()) # median approx at 0.05
if 'lapsetoprob' not in fix.keys():
    pars.add_param('lapsetoprob', 0, 1, gaussprob())


#%%
diff_data = pd.DataFrame([], columns=['accuracy', 'median RT'], 
                         index=subjects)
diff_post = pd.DataFrame([], columns=['accuracy', 'median RT'],
                         index = pd.MultiIndex.from_product(
                                 [subjects, np.arange(S)],
                                 names=['subject', 'sample']))
    
for sub in subjects:
    print('\rProcessing subject %2d ...' % sub)
    
    #%% load relevant options and results
    with pd.HDFStore(os.path.join(
            result, 's%02d_%s.h5' % (sub, stats)), 'r') as store:
        
        exclude_to = store['data_info']['exclude_to']
        censor_late = store['data_info']['censor_late']
        ndtdist = store['ndtdist'][0]
        
        psamples = store['parameters']
    
    data = helpers.load_subject(sub, exclude_to=exclude_to, 
                                censor_late=censor_late)
    
    sim, stat, data = snlsim.create_simulator(
            data, pars, stats, exclude_to, ndtdist, fix)
    
 
    #%% posterior predictive check of complete distribution
    # use only median to predict
    #pars_post = np.tile(np.median(all_ps[-1], axis=0)[None, :], (1000, 1))
    # use full posterior
    pars_post = psamples.loc[R]
    
    resp = sim.sim(pars_post)
    
    choices_post = resp[:, 0].reshape(sim.model.L, pars_post.shape[0], order='F')
    rts_post = resp[:, 1].reshape(sim.model.L, pars_post.shape[0], order='F')
    
    
    #%% check fit to differences in median RT and accuracy across difficulties
    diff_post.loc[sub, 'accuracy'], diff_post.loc[sub, 'median RT'] = (
            helpers.diff_diff(choices_post, rts_post, data.easy, 
                              sim.model.correct))
    
    dacc, dmed = helpers.diff_diff(
            data.response, data.RT, data.easy, sim.model.correct)
    diff_data.loc[sub] = np.r_[dacc, dmed]
    
print('done.')


#%% 
ddiff = diff_post.copy()
for sub in subjects:
    ddiff.loc[sub] = ddiff.loc[sub].values - diff_data.loc[sub].values
    
fig, axes = plt.subplots(1, 2)

for ax, st in zip(axes, ['accuracy', 'median RT']):
    for sub in subjects:
        ax.scatter(np.ones(S) * diff_data.loc[sub, st], diff_post.loc[sub, st],
                   alpha=0.05)
        
    xl = ax.get_xlim()
    ax.plot(xl, xl, 'k')
    
    ax.set_title(st)
    ax.set_xlabel('data value')
    
axes[0].set_ylabel('posterior predictive value')