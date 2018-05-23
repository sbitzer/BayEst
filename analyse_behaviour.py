#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:44:52 2018

@author: bitzer
"""

import helpers
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats


#%% load data
subjects = helpers.find_available_subjects()

alldata = pd.concat([helpers.load_subject(sub) for sub in subjects],
                    keys=subjects, names=['subject', 'trial'])


#%% number of data points per subject
fig, ax = plt.subplots()
alldata.RT.groupby('subject').count().plot.bar()

ax.set_ylabel('# data points')

              
#%% RT distributions
fig, ax = plt.subplots()

sns.violinplot(x="subject", y="RT", hue="easy", 
               data=alldata.reset_index('subject'), split=True, ax=ax);
ax.set_title('RT distributions across subjects')
               

#%% median RTs
medianRTs = pd.concat([alldata[alldata.easy].RT.groupby('subject').median(),
                       alldata[~alldata.easy].RT.groupby('subject').median()],
                      keys=['easy', 'hard'], names=['difficulty', 'subject'])

grid = sns.jointplot('easy', 'hard', data=medianRTs.unstack('difficulty'))

ax = grid.ax_joint
ax.set_autoscale_on(False)
ax.plot(ax.get_xlim(), ax.get_xlim(), '--k', zorder=0)
grid.ax_marg_x.set_title('median RTs across subjects')


#%% error rate
erate = pd.concat(
        [alldata[alldata.easy].error.groupby('subject').mean(),
         alldata[~alldata.easy].error.groupby('subject').mean()],
        keys=['easy', 'hard'], names=['difficulty', 'subject'])

grid = sns.jointplot('easy', 'hard', data=erate.unstack('difficulty'))

ax = grid.ax_joint
ax.set_autoscale_on(False)
ax.plot(ax.get_xlim(), ax.get_xlim(), '--k', zorder=0)
grid.ax_marg_x.set_title('error rates across subjects')


#%% error rate vs. median RT
grid = sns.jointplot('error rate', 'median RT', data=pd.concat(
        [erate, medianRTs], axis=1, keys=['error rate', 'median RT']))


#%% are fast trials more error prone than late trials?
#   within each subject and difficulty do median split on RT 
#   and compute error rate

erate_split = pd.Series(
        pd.np.zeros(2 * 2 * subjects.size),
        pd.MultiIndex.from_product(
                [['easy', 'hard'], subjects, ['fast', 'slow']], 
                names=['difficulty', 'subject', 'subset']), 
        name='error rate')

for sub in subjects:
    for easy in [True, False]:
        easystr = 'easy' if easy else 'hard'
        data = alldata.loc[sub]
        data = data[data.easy == easy]
        
        erate_split.loc[(easystr, sub, 'fast')] = data[
                data.RT < medianRTs.loc[(easystr, sub)]].error.mean()
        
        erate_split.loc[(easystr, sub, 'slow')] = data[
                data.RT >= medianRTs.loc[(easystr, sub)]].error.mean()

erate_split = erate_split.unstack('subset')

grid = sns.jointplot('fast', 'slow', data=erate_split)

ax = grid.ax_joint
ax.set_autoscale_on(False)
ax.plot(ax.get_xlim(), ax.get_xlim(), '--k', zorder=0)
grid.ax_marg_x.set_title('error rates across subjects and difficulty')

fig, ax = plt.subplots()
erate_diff = erate_split.fast - erate_split.slow
sns.distplot(erate_diff, rug=True, ax=ax)

ax.set_xlabel('error rate difference (fast - slow)')

tval, pval = scipy.stats.ttest_1samp(erate_diff, 0)
ax.set_title('t = %5.2f, p = %6.4f' % (tval, pval))