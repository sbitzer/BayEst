#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:51:18 2018

@author: bitzer
"""

from __future__ import print_function, division

import numpy as np
import pandas as pd
import rotated_directions as rtd
import helpers

import matplotlib.pyplot as plt
import seaborn as sns

#%% helper functions
def print_performance(ch, rt, correct, prefix=''):
    if prefix == '':
        prefix = []
    else:
        prefix = [prefix]
        
    print(' '.join(prefix + ['accuracy : {:4.1f}%'.format(
            (ch == correct).mean() * 100)]))
    
    print(' '.join(prefix + ['median RT: {:4.2f} s'.format(
            np.median(rt))]))


#%% load data of one subject and create corresponding model
data = helpers.load_subject(19)

model = rtd.rotated_directions_diff(
        {'directions': data.tarDir, 'criteria': data.critDir}, dt=helpers.dt, 
        maxrt=2., choices=[1, -1], ndtdist='uniform')


#%% set some sensible parameters
model.diffstd = 1000
model.cpsqrtkappa = 1
model.critstd = 0.7
model.dirstd = 0.1
model.cnoisestd = 1e-12
model.dnoisestd = 1e-12
model.bound = 0.7
model.lapseprob = 0

model.ndtloc = -12
model.ndtspread = 0
#model.ndtloc = np.log(0.4)
#model.ndtspread = 0.4


#%% trial selections
# both trials with same rotation from criterion, but different criteria
crittrials = np.r_[9, 1923]

crittype = data.loc[crittrials, 'critDir'].map(
        {0:'cardinal', 45:'oblique', 90:'cardinal', 135:'oblique'})


#%%
fig, ax = model.plot_example_timecourses(crittrials, dvtype='logprobdiff')

lines = ax.get_children()[:crittrials.size]
lines[0].set_color('C3')
ax.legend(lines, crittype.values)


#%% investigate the interaction between cpsqrtkappa and critstd on differences
#   in choice behaviour for crittrials

# select some parameter values for std
P = 100
vals = np.logspace(np.log10(0.01), np.log10(8), P)
params = np.c_[1 / np.tile(vals, P), np.repeat(vals, P)]
parnames = ['cpsqrtkappa', 'critstd']

parindex = pd.MultiIndex.from_arrays(
        [params[:, 0], params[:, 1]], names=parnames)

# generated responses from model
def gen_responses(trial):
    ch, rt = model.gen_response_with_params(trial, params, parnames)
    return pd.DataFrame(np.c_[ch, rt], index=parindex, columns=['choice', 'RT'])

responses = pd.concat([gen_responses(tr) for tr in crittrials],
                      axis=1, keys=crittrials, names=['trial', 'response'])

rtdiff = responses[(crittrials[0], 'RT')] - responses[(crittrials[1], 'RT')]
rtdiff = rtdiff.unstack('critstd')


#%% plot result
edges = lambda vals: np.r_[vals[0] - (vals[1] - vals[0]), 
                           (vals[1:] + vals[:-1])/2, 
                           vals[-1] + (vals[-1] - vals[-2])]
xedges = edges(rtdiff.columns)
yedges = edges(rtdiff.index)

isvalid_ch = ~(  (responses[(crittrials[0], 'choice')] == 1) 
               & (responses[(crittrials[1], 'choice')] == 1))
isvalid_ch = isvalid_ch.unstack('critstd')
mask = lambda a: np.ma.array(a, mask=isvalid_ch)

fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)

qmesh = axes[0].pcolormesh(xedges, yedges, mask(rtdiff),
                           cmap='RdBu', vmin=-0.3, vmax=0.3)
fig.colorbar(qmesh, ax=axes[0])
axes[0].set_title('RT difference (tr[0] - tr[1]')

qmesh = axes[1].pcolormesh(xedges, yedges, 
                           mask(responses[(crittrials[1], 'RT')].unstack('critstd')),
                           vmin=0, vmax=2)
fig.colorbar(qmesh, ax=axes[1])
axes[1].set_title('RT (tr[1])')

qmesh = axes[2].pcolormesh(xedges, yedges, 
                           responses[(crittrials[0], 'choice')].unstack('critstd'),
                           cmap='RdBu', vmin=-1.5, vmax=1.5)
axes[2].set_title('choice (tr[0])')
qmesh = axes[3].pcolormesh(xedges, yedges, 
                           responses[(crittrials[1], 'choice')].unstack('critstd'),
                           cmap='RdBu', vmin=-1.5, vmax=1.5)
axes[3].set_title('choice (tr[1])')

axes[0].set_xlabel(rtdiff.columns.name)
axes[0].set_ylabel('cpstd')


#%%
model.cpsqrtkappa = 1
model.critstd = 0.7
model.dirstd = 0.1

times, lpD, lpCR, lCR, lE_OM, lE_DR, lp_rot = model.gen_timecourses(crittrials)


#%%
radcrit = rtd.to_rad(model.criteria)
cps = .2 ** 2
lpcr = rtd.logsumexp_2d(np.c_[
                np.cos(2 * radcrit), 
                np.cos(2 * (radcrit - np.pi / 2)),
                ] * cps, axis=1)[:, 0]


#%% generate responses from the model
ch, rt = model.gen_response(np.arange(model.L))

# basic performance measures
print('\nbasic performance measures')
print('--------------------------')
print_performance(ch, rt, model.correct)
print_performance(ch[data.easy], rt[data.easy], model.correct[data.easy], 'easy')
print_performance(ch[~data.easy], rt[~data.easy], model.correct[~data.easy], 'hard')


#%% RT distributions
fig = plt.figure()
sns.distplot(data.RT[data.easy], kde=False, label='data easy')
sns.distplot(data.RT[~data.easy], kde=False, label='data hard')
ax = sns.distplot(rt[data.easy], kde=False, label='easy')
ax = sns.distplot(rt[~data.easy], kde=False, label='hard')
#for df in model.differences:
#    ax = sns.distplot(rt[data.difference == df], kde=False, label=str(df))

ax.legend()