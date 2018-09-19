#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:01:37 2018

@author: bitzer
"""

import numpy as np
import matplotlib.pyplot as plt

import helpers
from rotated_directions import rotated_directions as dirmodel


#%% create model based on experimental data
data = helpers.load_subject(19, exclude_to=False, censor_late=True)
N = data.shape[0]

choices = [1, -1]
ndtdist = 'uniform'
model = dirmodel(
        data.tarDir / 180. * np.pi, helpers.dt, 
        data.tarDir.unique() / 180. * np.pi, data.critDir / 180. * np.pi, 
        maxrt=helpers.maxrt + helpers.dt, toresponse=helpers.toresponse, 
        choices=choices, ndtdist=ndtdist)

model.bound = 0.7
model.intstd = 0.18
model.noisestd = 1e-7


#%% plot example model dynamics
seed = 48291043
fig, ax = model.plot_example_timecourses(
        np.arange(20), dvtype='logprobdiff', seed=seed)


#%% create and simulate from model
choices, rts = model.gen_response(np.arange(N))

plt.figure()
model.plot_response_distribution(choices, rts)

is_correct = choices == model.correct
is_correct.mean()


#%% log-probability increments increase as more evidence is sampled
times, logprob_cw, logprob, logpost, logliks = model.gen_timecourses(
        np.arange(N))

fig, axes = plt.subplots(1, 4, sharex=True, figsize=(12, 4))

axes[0].plot(times[1:], np.abs(logliks[:, 0, :]).mean(axis=1), label='loglik')
axes[0].set_title('|log-lik|')

diffs = np.diff(logprob[:, 0, :], axis=0)
axes[1].plot(times[1:], np.abs(diffs).mean(axis=1), label='logprob-diff')
axes[1].set_title('|log prob diff| (dir 0)')

diffs = np.diff(logprob_cw, axis=0)
axes[2].plot(times[1:], np.abs(diffs).mean(axis=1), label='logp-diff')
axes[2].set_title('|log prob diff| (cw)')

diffs = np.diff(logprob_cw - np.log(1 - np.exp(logprob_cw)), axis=0)
axes[3].plot(times[1:], np.abs(diffs).mean(axis=1), label='logp-diff')
axes[3].set_title('|LLR diff|')

for ax in axes:
    ax.set_xlabel('time (s)')

fig.tight_layout()


#%% check posterior probability densities/distribs over directions across time
P = 200
cols = plt.cm.Blues(np.linspace(0, 1, P))

# directions are unsorted in model so find order for displaying densities
sortind = np.argsort(model.directions)

fig, ax = plt.subplots()

for lp, col, p in zip(logprob[:P, :, 0], cols, range(P)):
    ax.plot(np.exp(lp[sortind]), color=col, label='t={}'.format(p * model.dt))
#    ax.plot(lp[sortind], color=col, label='t={}'.format(p * model.dt))
    
#ax.legend()