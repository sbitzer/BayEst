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

import matplotlib.pyplot as plt


#%% 
stats = 'hist'

result = os.path.join(helpers.resultsdir, 
                      'snl/rotated_directions/201810090117')

comparison_result = os.path.join(
        helpers.resultsdir, 'snl/rotated_directions/201807061824')

subjects = [int(os.path.basename(f)[1:3]) 
            for f in glob(os.path.join(result, '*%s.log' % stats))]


#%%
with pd.HDFStore(os.path.join(
        result, 's%02d_%s.h5' % (subjects[0], stats)), 'r') as store:
    fix = store['fix']
    psamples = store['parameters']
    R = psamples.index.get_level_values('round').unique().size
    S = psamples.loc[R].shape[0]


#%%
diff_data = pd.DataFrame([], columns=['accuracy', 'median RT'], 
                         index=subjects)
diff_post = pd.DataFrame([], columns=['accuracy', 'median RT'],
                         index = pd.MultiIndex.from_product(
                                 [subjects, np.arange(S)],
                                 names=['subject', 'sample']))
    
for sub in subjects:
    print('\rProcessing subject %2d ...' % sub)
    
    #%% generate posterior predictive data
    choices_post, rts_post, data, model = (
            snlsim.generate_posterior_predictive_data(result, sub, stats))
    
    if len(subjects) == 1 and comparison_result:
        choices_post_comp, rts_post_comp, data_comp, model_comp = (
            snlsim.generate_posterior_predictive_data(
                    comparison_result, sub, stats))
    
    
    #%% check fit to differences in median RT and accuracy across difficulties
    diff_post.loc[sub, 'accuracy'], diff_post.loc[sub, 'median RT'] = (
            helpers.diff_diff(choices_post, rts_post, data.easy, 
                              model.correct))
    
    dacc, dmed = helpers.diff_diff(
            data.response, data.RT, data.easy, model.correct)
    diff_data.loc[sub] = np.r_[dacc, dmed]
    
print('done.')


#%% 
if len(subjects) == 1:
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
    
    is_cardinal = lambda deg: deg == 0 | deg == 90
    
    for ax, cond, label in zip(
            axes, [data.easy, data.critDir.map(is_cardinal)], 
            ['easy - hard', 'cardinal - oblique']):
        ax = helpers.plot_diffs(
                choices_post, rts_post, cond, model.correct, data, 
                label=label, ax=ax)
    
    if comparison_result:
        for ax, cond, label in zip(
                axes, [data_comp.easy, data_comp.critDir.map(is_cardinal)], 
                ['easy - hard', 'cardinal - oblique']):
            ax = helpers.plot_diffs(
                    choices_post_comp, rts_post_comp, cond, model_comp.correct, 
                    data_comp, label=label, ax=ax)
    

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