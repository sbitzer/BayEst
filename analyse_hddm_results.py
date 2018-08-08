#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:08:23 2018

@author: bitzer
"""
import numpy as np
import pandas as pd
import os

import helpers

import matplotlib.pyplot as plt
import seaborn as sns

#%% 
result = 'hddm_result_201807101804.h5'
result = os.path.join(helpers.resultsdir, 'hddm', result)

with pd.HDFStore(result, 'r') as store:
    samples = store['samples']
    S = store['scalar_opt'].PS
    censor_late = store['scalar_opt'].censor_late
    exclude_to = store['scalar_opt'].exclude_to
    
subjects = samples.index.get_level_values('subject').unique()


#%%
diff_data = pd.DataFrame([], columns=['accuracy', 'median RT'], 
                         index=subjects)
diff_post = pd.DataFrame([], columns=['accuracy', 'median RT'],
                         index = pd.MultiIndex.from_product(
                                 [subjects, np.arange(S)],
                                 names=['subject', 'sample']))

for sub in subjects:
    print('\rprocessing subject %2d ...' % sub, end='')
    data = helpers.load_subject(sub, exclude_to=exclude_to, 
                                censor_late=censor_late)
    stim = data.response.copy().astype(int)
    err = data.error.astype(bool)
    stim[err] = -stim[err]
    
    with pd.HDFStore(result, 'r') as store:
        ppc_data = store.select('ppc_data', 'subject=sub').loc[sub]
        
    ppc_data.sort_index(inplace=True)
    N = ppc_data.loc[0].shape[0]
    
    resh = lambda data: data.values.reshape(N, S, order='F')
        
    diff_post.loc[sub, 'accuracy'], diff_post.loc[sub, 'median RT'] = (
                helpers.diff_diff(resh(ppc_data.response), resh(ppc_data.rt), 
                                  ppc_data.loc[0].easy, ppc_data.loc[0].stim))
    
    dacc, dmed = helpers.diff_diff(
            data.response, data.RT, data.easy, stim)
    diff_data.loc[sub] = np.r_[dacc, dmed]

print('done.')

#%%
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


#%% plot difference in mean posterior drift rate vs. data median RT and acc
meanpar = samples.groupby('subject').mean()

# difference easy - hard in drifts
diff_data['v'] = meanpar['v(True)'] - meanpar['v(False)']
diff_data = diff_data.astype(float)

fig, axes = plt.subplots(1, 2, sharey=True)

for ax, st in zip(axes, ['accuracy', 'median RT']):
    sns.regplot(st, 'v', data=diff_data, ax=ax)
    ax.plot(ax.get_xlim(), [0, 0], 'k--')