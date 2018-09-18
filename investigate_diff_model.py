#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:51:18 2018

@author: bitzer
"""

from __future__ import print_function, division

import numpy as np
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


#%% generate responses from the model
model.diffstd = 1000
model.cpsqrtkappa = 2.2
model.critstd = 0.1
model.dirstd = 0.2
model.cnoisestd = 1e-12
model.dnoisestd = 0.4
model.lapseprob = 0

#model.ndtloc = -12
#model.ndtspread = 0
model.ndtloc = np.log(0.4)
model.ndtspread = 0.4

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