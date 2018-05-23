#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:59:24 2018

@author: bitzer
"""

import numpy as np
#import pandas as pd
import helpers

import rotated_directions as rtmodels

import pyEPABC
from pyEPABC.parameters import exponential, gaussprob
import matplotlib.pyplot as plt


#%% load data
sub = 18
data = helpers.load_subject(sub)


#%% create model
dt = 0.05

# first of these indicates clockwise rotation, second anti-clockwise
choices = [1, -1]

model = rtmodels.rotated_directions(
        data.tarDir / 180 * np.pi, dt, data.tarDir.unique() / 180 * np.pi, 
        data.critDir / 180 * np.pi, maxrt=data.dropna().RT.max() + dt,
        toresponse=helpers.toresponse, choices=choices)

errorm = model.correct != data.response
ind = data.response != 0
assert np.all(data[ind].error == errorm[ind]), ("Model and data don't agree "
             "about which trials are error trials!")


#%% infer
pars = pyEPABC.parameters.parameter_container()
pars.add_param('noisestd', 0, 1, exponential())
pars.add_param('intstd', 0, 1, exponential())
pars.add_param('bound', 0, 1, gaussprob(width=0.5, shift=0.5))
pars.add_param('bias', 0, .2, gaussprob())
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
        data[['response', 'RT']].values, simfun, None, pars.mu, pars.cov, 
        epsilon=epsilon, minacc=2000, samplestep=10000, samplemax=6000000, 
        npass=2, alpha=0.5, veps=veps)


#%% show posterior distribution
pg = pars.plot_param_dist(ep_mean, ep_cov)
pg.fig.tight_layout()

pg.fig.savefig('%02d_posterior.svg' % sub)


#%% posterior predictive check
choices_post, rts_post = model.gen_response_from_Gauss_posterior(
        np.arange(model.L), pars.names, ep_mean, ep_cov, 30, pars.transform)

rtfig, rtaxes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 5))

for row, rtdists in zip(rtaxes, ((
        (data.response, data.RT), 
        (data[data.easy].response, data[data.easy].RT), 
        (data[~data.easy].response, data[~data.easy].RT)), (
        (choices_post, rts_post), 
        (choices_post[data.easy], rts_post[data.easy]), 
        (choices_post[~data.easy], rts_post[~data.easy])))):
    for ax, rtdist in zip(row, rtdists):
        model.plot_response_distribution(rtdist[0], rtdist[1], ax=ax)
        ax.plot(np.median(rtdist[1]) * np.r_[1, 1], np.r_[0, 1.5], 'k')

rtaxes[0, 0].set_title('all trials')
rtaxes[0, 1].set_title('easy trials')
rtaxes[0, 2].set_title('hard trials')

rtfig.savefig('%02d_easy_vs_hard.svg' % sub)


#%% show dt vs ndt distributions
ndtfig, ndtax = model.plot_dt_ndt_distributions(ep_mean, ep_cov, pars)

ndtfig.savefig('%02d_dt_vs_ndt.svg' % sub)