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
import seaborn as sns


#%% 
stats = 'hist'

result = os.path.join(helpers.resultsdir, 
                      'snl/rotated_directions/201907240843')

comparison_result = os.path.join(
        helpers.resultsdir, 'snl/rotated_directions/201807061824')

subjects = [int(os.path.basename(f)[1:3]) 
            for f in glob(os.path.join(result, '*%s.log' % stats))]
subjects = sorted(subjects)[:-1]


#%%
with pd.HDFStore(os.path.join(
        result, 's%02d_%s.h5' % (subjects[0], stats)), 'r') as store:
    fix = store['fix']
    psamples = store['parameters']
    R = psamples.index.get_level_values('round').unique().size
    S = psamples.loc[R].shape[0]


#%% load results and prepare measures
diff_data = pd.DataFrame([], columns=['accuracy', 'median RT'], 
                         index=subjects, dtype=float)
diff_post = pd.DataFrame([], columns=['accuracy', 'median RT'],
                         index = pd.MultiIndex.from_product(
                                 [subjects, np.arange(S)],
                                 names=['subject', 'sample']), dtype=float)
diff_post_comp = pd.DataFrame(
        [], columns=['accuracy', 'median RT'],
        index = pd.MultiIndex.from_product(
                [subjects, np.arange(S)], names=['subject', 'sample']), 
        dtype=float)
    
loglik = []
loglik_comp = []
for sub in subjects:
    print('\rProcessing subject %2d ...' % sub)
    
    with pd.HDFStore(os.path.join(
            result, 's{:02d}_{}.h5'.format(sub, stats)), 'r') as store:
        loglik.append(store['logdens']['loglik'].loc[R])
    
    # generate posterior predictive data
    choices_post, rts_post, data, model = (
            snlsim.generate_posterior_predictive_data(result, sub, stats))
    
    # check fit to differences in median RT and accuracy across difficulties
    diff_post.loc[sub, 'accuracy'], diff_post.loc[sub, 'median RT'] = (
            helpers.diff_diff(choices_post, rts_post, data.easy, 
                              model.correct))
    
    if comparison_result:
        with pd.HDFStore(os.path.join(
                comparison_result, 's{:02d}_{}.h5'.format(sub, stats)), 'r') as store:
            loglik_comp.append(store['logdens']['loglik'].loc[R])
        
        choices_post_comp, rts_post_comp, data_comp, model_comp = (
            snlsim.generate_posterior_predictive_data(
                    comparison_result, sub, stats))
        diff_post_comp.loc[sub, 'accuracy'], diff_post_comp.loc[sub, 'median RT'] = (
            helpers.diff_diff(choices_post_comp.reindex(data.index),
                              rts_post_comp.reindex(data.index), data.easy, 
                              model.correct))
    
    dacc, dmed = helpers.diff_diff(
            data.response, data.RT, data.easy, model.correct)
    diff_data.loc[sub] = np.r_[dacc, dmed]
    
print('done.')

loglik = pd.concat(loglik, axis=1)
loglik.columns = pd.Index(subjects, name='subject')
loglik_comp = pd.concat(loglik_comp, axis=1)
loglik_comp.columns = pd.Index(subjects, name='subject')


#%% log-likelihoods of posterior samples
fun = lambda x: x.stack().reset_index('subject')
logliks = pd.concat(
        [fun(loglik), fun(loglik_comp)], axis=0, 
        keys=['selected', 'comparison'], 
        names=['result', 'sample']).reset_index('result')

logliks.columns = ['result', 'subject', 'loglik']

sns.catplot(x="subject", y='loglik', hue="result",
            kind="box", data=logliks);

fig, ax = plt.subplots()
sns.distplot(loglik.median() - loglik_comp.median(), rug=True, ax=ax)
ax.set_xlabel('median loglik difference (new - old)')


#%% estimate goodness of fit directly on summary statistics
within_deviance = lambda x: np.sum(x**2, axis=1)
within_deviance = lambda x: np.median(np.abs(x), axis=1)

across_deviance = lambda x: np.mean(np.abs(x), axis=0)

fit, eh = snlsim.estimate_posterior_fit(
        result, subjects, stats, within_deviance, across_deviance)

fig, ax = plt.subplots()
sns.distplot(fit, rug=True, ax=ax, label='selected')

if comparison_result:
    fit_comp, eh_comp = snlsim.estimate_posterior_fit(
            comparison_result, subjects, stats, within_deviance, 
            across_deviance)
    sns.distplot(fit_comp, rug=True, ax=ax, label='comparison')

ax.legend()


#%% check fit of differences in accuracy and median RT across easy/hard conditions
fig, axes = plt.subplots(1, 2)

for measure, ax in zip(['accuracy', 'median RT'], axes):
    sns.distplot(
            (diff_data[measure] 
            - diff_post[measure].groupby('subject').median()).dropna(),
            label='selected', ax=ax, bins=12)
    sns.distplot(
            (diff_data[measure] 
            - diff_post_comp[measure].groupby('subject').median()).dropna(), 
            label='comparison', ax=ax, bins=12)
    
    ax.set_xlabel(measure)
    ax.legend()


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