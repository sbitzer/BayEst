#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:01:37 2018

@author: bitzer
"""

import rotated_directions as rtmodels
import numpy as np
import matplotlib.pyplot as plt


#%% create and simulate from model
C = 6
D = 12
N = 1000

directions = np.linspace(0, 2 * np.pi - 2 * np.pi / D, D)
Trials = directions[np.random.randint(D, size=N)]
criteria = np.linspace(0, np.pi - np.pi / C, C)
criteria = criteria[np.random.randint(C, size=N)]

dt = 0.05

model = rtmodels.rotated_directions(Trials, dt, directions, criteria, maxrt=2)

model.noisestd = 1
model.intstd = 0.7
model.bound = 0.51
model.lapseprob = 0
choices, rts = model.gen_response(np.arange(N))

plt.figure()
model.plot_response_distribution(choices, rts)

is_correct = choices == model.correct
is_correct.mean()

fig, ax = model.plot_example_timecourses(np.arange(min(N, 200)))


#%% log-probability increments increase as more evidence is sampled
times, logprob_cw, logprob, logpost, logliks = model.gen_timecourses(
        np.arange(N))

fig, axes = plt.subplots(1, 3, sharex=True, figsize=(10, 4))

axes[0].plot(times[1:], np.abs(logliks[:, 0, :]).mean(axis=1), label='loglik')
axes[0].set_title('|log-lik|')

diffs = np.diff(logprob[:, 0, :], axis=0)
axes[1].plot(times[1:], np.abs(diffs).mean(axis=1), label='logprob-diff')
axes[1].set_title('|log prob diff| (dir 0)')

diffs = np.diff(logprob_cw, axis=0)
axes[2].plot(times[1:], np.abs(diffs).mean(axis=1), label='logp-diff')
axes[2].set_title('|log prob diff| (cw)')

for ax in axes:
    ax.set_xlabel('time (s)')

fig.tight_layout()