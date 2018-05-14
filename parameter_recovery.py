#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 15:53:00 2018

@author: bitzer
"""

import rotated_directions as rtmodels
import pyEPABC
from pyEPABC.parameters import exponential, gaussprob
import numpy as np
import matplotlib.pyplot as plt


#%% create trial structure for simulated experiment
# number of repetitions of the same trial type
R = 40

# possible criterion orientations
criteria = np.r_[0, 45, 90, 135]
C = criteria.size

# trials are generated by combining each criterion orientation with a motion
# direction that is either 15 or 30 degrees rotated clockwise or anti-clockwise
Trials = criteria[None, :] + np.r_[-30, -15, 15, 30][:, None]
# motion directions can be flipped
Trials = np.concatenate([Trials, Trials+180], axis=0) % 360

# unique motion directions that occurred in the experiment
directions = np.unique(Trials) / 180 * np.pi

# which are the easy trials? (30 degree rotations)
easy = np.tile(np.r_[True, False, False, True][:, None], (2 * R, C))
easy = easy.flatten()

# bring everything into trial-wise format understood by model
Trials = np.tile(Trials, [R, 1]).flatten() / 180 * np.pi
criteria = np.tile(criteria, (R * 8, 1)).flatten() / 180 * np.pi

N = Trials.size


#%% create model and sample data from it
dt = 0.05

model = rtmodels.rotated_directions(Trials, dt, directions, criteria, maxrt=2)

model.noisestd = 0.5
model.intstd = 0.3
model.bound = 0.8
model.lapseprob = 0.05
model.ndtmean = np.log(0.2)
model.ndtspread = 0.6

choices, rts = model.gen_response(np.arange(N))

fig = plt.figure()
model.plot_response_distribution(choices, rts)

is_correct = choices == model.correct
print('accuracies: %4.2f (overall), %4.2f (easy), %4.2f (hard)' % (
        is_correct.mean(), is_correct[easy].mean(), is_correct[~easy].mean()))
print('median RTs: %4.2f (overall), %4.2f (easy), %4.2f (hard)' % (
        np.median(rts), np.median(rts[easy]), np.median(rts[~easy])))


#%% try to recover the parameters with pyEPBAC
pars = pyEPABC.parameters.parameter_container()
pars.add_param('noisestd', 0, 1, exponential())
pars.add_param('intstd', 0, 1, exponential())
pars.add_param('bound', 0, 1, gaussprob(width=0.5, shift=0.5))
pars.add_param('ndtmean', -2, 1)
pars.add_param('ndtspread', np.log(0.2), 1, exponential())
pars.add_param('lapseprob', -1.65, 1, gaussprob()) # median approx at 0.05
pars.add_param('lapsetoprob', 0, 1, gaussprob())

#pars.plot_param_dist()

simfun = lambda data, dind, parsamples: model.gen_distances_with_params(
            data[0], data[1], dind, pars.transform(parsamples), pars.names)

epsilon = 0.05
veps = 2 * epsilon

# calling EPABC:
ep_mean, ep_cov, ep_logml, nacc, ntotal, runtime = pyEPABC.run_EPABC(
        np.c_[choices, rts], simfun, None, pars.mu, pars.cov, 
        epsilon=epsilon, minacc=2000, samplestep=10000, samplemax=6000000, 
        npass=2, alpha=0.5, veps=veps)


#%% compare inferred and true parameters
# prior vs. posterior pdfs
fig, axes = pars.compare_pdfs(ep_mean, ep_cov, figsize=[10, 6])

# plot the true values
for name, ax in zip(pars.names.values, axes.flatten()[:pars.P]):
    ax.plot(getattr(model, name), 0, '*k', label='true value')

axes[0, 0].legend()

fig.savefig('recovery_result.svg')


#%% show posterior distribution
pg = pars.plot_param_dist(ep_mean, ep_cov)
pg.fig.tight_layout()
pg.fig.savefig('posterior_params.svg')


#%% posterior predictive check
choices_post, rts_post = model.gen_response_from_Gauss_posterior(
        np.arange(N), pars.names, ep_mean, ep_cov, 100, pars.transform)

rtfig, rtaxes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 5))

for row, rtdists in zip(rtaxes, ((
        (choices, rts), 
        (choices[easy], rts[easy]), 
        (choices[~easy], rts[~easy])), (
        (choices_post, rts_post), 
        (choices_post[easy], rts_post[easy]), 
        (choices_post[~easy], rts_post[~easy])))):
    for ax, rtdist in zip(row, rtdists):
        model.plot_response_distribution(rtdist[0], rtdist[1], ax=ax)
        ax.plot(np.median(rtdist[1]) * np.r_[1, 1], np.r_[0, 1.5], 'k')

rtaxes[0, 0].set_title('all trials')
rtaxes[0, 1].set_title('easy trials')
rtaxes[0, 2].set_title('hard trials')

rtfig.savefig('posterior_predictive.svg')


#%% check the decision time distributions
ndtmean = model.ndtmean
model.ndtmean = -100
choices_dt, rts_dt = model.gen_response(np.arange(N), 10)
choices_dt = choices_dt.T
rts_dt = rts_dt.T
model.ndtmean = ndtmean

mu_dec = ep_mean.copy()
ndind = pars.names[pars.names.map(lambda name: name == 'ndtmean')].index
mu_dec[ndind] = -100
cov_dec = ep_cov.copy()
cov_dec[ndind, ndind] = 1e-10
choices_dt_post, rts_dt_post = model.gen_response_from_Gauss_posterior(
        np.arange(N), pars.names, mu_dec, cov_dec, 10, pars.transform)

rtfig, rtaxes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 5))

for row, rtdists in zip(rtaxes, ((
        (choices_dt, rts_dt), 
        (choices_dt[easy], rts_dt[easy]), 
        (choices_dt[~easy], rts_dt[~easy])), (
        (choices_dt_post, rts_dt_post), 
        (choices_dt_post[easy], rts_dt_post[easy]), 
        (choices_dt_post[~easy], rts_dt_post[~easy])))):
    for ax, rtdist in zip(row, rtdists):
        model.plot_response_distribution(rtdist[0], rtdist[1], ax=ax)
        ax.plot(np.median(rtdist[1]) * np.r_[1, 1], np.r_[0, 1.5], 'k')

rtaxes[0, 0].set_title('all trials')
rtaxes[0, 1].set_title('easy trials')
rtaxes[0, 2].set_title('hard trials')

rtfig.savefig('posterior_predictive_dtonly.svg')