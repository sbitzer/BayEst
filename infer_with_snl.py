#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:00:24 2018

@author: bitzer
"""
#%% imports
import numpy as np
import pandas as pd
import os

import helpers
import snl_simulators as snlsim

import snl
import pyEPABC.parameters as parameters
from pyEPABC.parameters import exponential, gaussprob, zero

import matplotlib.pyplot as plt


#%% define parameters and their prior
ndtdist = 'uniform'
fix = {'bound': 0.7, 'diffstd': 1000, 'cnoisestd': 1e-12,
       'cpsqrtkappa': 0}

pars = parameters.parameter_container()
if 'diffstd' not in fix.keys():
    pars.add_param('diffstd', 0, 10, exponential())
if 'cpsqrtkappa' not in fix.keys():
    pars.add_param('cpsqrtkappa', 1, 1, zero())
if 'critstd' not in fix.keys():
    pars.add_param('critstd', 0, 1, exponential())
if 'cnoisestd' not in fix.keys():
    pars.add_param('cnoisestd', 0, 1, exponential())
if 'dnoisestd' not in fix.keys():
    pars.add_param('dnoisestd', 0, 1.2, exponential())
if 'dirstd' not in fix.keys():
    pars.add_param('dirstd', 0, 1.2, exponential())
if 'bound' not in fix.keys():
    pars.add_param('bound', 0, 1, gaussprob(width=0.5, shift=0.5))
if 'bias' not in fix.keys():
    pars.add_param('bias', 0, .2, gaussprob())
if 'ndtloc' not in fix.keys():
    pars.add_param('ndtloc', -2, 1)
if 'ndtspread' not in fix.keys():
    pars.add_param('ndtspread', np.log(0.2), 1, exponential())
if 'lapseprob' not in fix.keys():
    pars.add_param('lapseprob', -1.65, 1, gaussprob()) # median approx at 0.05
if 'lapsetoprob' not in fix.keys():
    pars.add_param('lapsetoprob', 0, 1, gaussprob())

prior = snl.pdfs.Gaussian(m=pars.mu, S=pars.cov)


#%%
#subjects = helpers.find_available_subjects()
subjects = [19]

censor_late = True
exclude_to = False

stats = 'hist'

resdir = os.path.join(helpers.resultsdir, 'snl', 'rotated_directions', 
                      pd.datetime.now().strftime('%Y%m%d%H%M'))
os.mkdir(resdir)

for sub in subjects:
    data = helpers.load_subject(sub, exclude_to=exclude_to, 
                                censor_late=censor_late)
    
    if stats == 'id':
        conditions = data.easy
    else:
        conditions = None
    
    sim, stat, data = snlsim.create_simulator(
            data, pars, stats, exclude_to, ndtdist, fix)
    
#    p = pars.sample(10)
#    stat.calc(sim.sim(p))
    
    obs_xs = stat.calc(data[['response', 'RT']])


#%%
    fbase = os.path.join(resdir, 's%02d_' % sub + stats)
    snl.run_snl(obs_xs, sim, stat, prior, filebase=fbase, conditions=conditions)
    
    with pd.HDFStore(fbase + '.h5', 'r+') as store:
        # save extra information about this result
        store['fix'] = pd.Series(fix)
        store['prior_mu'] = pd.Series(pars.mu, pars.names)
        store['prior_cov'] = pd.DataFrame(pars.cov, columns=pars.names, 
                                          index=pars.names)
        store['data_info'] = pd.Series(
                [sub, censor_late, exclude_to], 
                index=['subject', 'censor_late', 'exclude_to'])
        store['stats'] = pd.Series(stats)
        store['ndtdist'] = pd.Series(ndtdist)
        
        if stats == 'hist':
            store['stats_opt'] = pd.Series(stat.bins)
        elif stats == 'quant':
            store['stats_opt'] = pd.Series(stat.percentiles)


#%% show diagnostics
if len(subjects) == 1:
    with pd.HDFStore(fbase + '.h5', 'r') as store:
        # load stored results
        psamples = store['parameters']
        xsamples = store['simdata']
        logdens = store['logdens']
        trainlog = store['trainlog']
        elapsed = store['elapsed']
    
    print('elapsed: %s' % elapsed.loc['elapsed'])

    R = trainlog.index.get_level_values('round').unique().size


    #%% plot some training diagnostics 
    fig, axes = plt.subplots(1, 2)
    
    for dens, ax in zip(logdens.columns, axes):
        ax.plot(logdens[dens].values)
        ax.set_ylabel(dens)
        ax.set_xlabel('sample')
        
        
    fig, ax = plt.subplots()
    ax.plot(trainlog.val_loss.values)
    ax.set_xlabel('epoch')
    ax.set_ylabel('validation loss')
    
    
    #%% stats for posterior parameters
    ptr = pd.DataFrame(pars.transform(psamples.loc[R].values), 
                       columns=psamples.columns)
    pnames = sorted(list(set(pars.names).intersection(set(
            ['noisestd', 'intstd', 'bound', 'ndtloc', 'lapseprob']))))
    print(ptr[pnames].describe([0.05, 0.5, 0.95]))
    
    
    #%% posterior parameters
    pg = pars.plot_param_hist(psamples.loc[R].values)
    
    # ignore correlations and estimate most likely marginal posterior values
    def get_mode(a):
        cnt, bins = np.histogram(a, 'auto')
        ind = cnt.argmax()
        return bins[ind:ind+2].mean()
    
    modes = np.array([get_mode(psamples.loc[R, name]) for name in pars.names])
    modes_tr = pd.Series(pars.transform(modes[None, :])[0], index=psamples.columns)
    
    for ax, mode, median in zip(pg.diag_axes, modes_tr, ptr.median()):
        ax.plot(mode, 0.1 * ax.get_ylim()[1], '*k')
        ax.plot(median, 0.05 * ax.get_ylim()[1], '*g')
    
    
    #%% check fit to summary statistics for samples in last round
    fig, ax = plt.subplots()
    ax.boxplot(xsamples.loc[R].values)
    ax.plot(np.arange(obs_xs.size) + 1, obs_xs, '*', color='C0')
    
    
    #%% posterior predictive check of complete distribution
    # use only median to predict
    #pars_post = np.tile(np.median(all_ps[-1], axis=0)[None, :], (1000, 1))
    # use full posterior
    pars_post = psamples.loc[R]
    
    resp = sim.sim(pars_post)
    
    choices_post = resp[:, 0].reshape(sim.model.L, pars_post.shape[0], order='F')
    rts_post = resp[:, 1].reshape(sim.model.L, pars_post.shape[0], order='F')
    
    rtfig, rtaxes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 5))
    
    for row, rtdists in zip(rtaxes, ((
            (data.response, data.RT), 
            (data[data.easy].response, data[data.easy].RT), 
            (data[~data.easy].response, data[~data.easy].RT)), (
            (choices_post, rts_post), 
            (choices_post[data.easy], rts_post[data.easy]), 
            (choices_post[~data.easy], rts_post[~data.easy])))):
        for ax, rtdist in zip(row, rtdists):
            sim.model.plot_response_distribution(rtdist[0], rtdist[1], ax=ax)
            ax.plot(np.median(rtdist[1]) * np.r_[1, 1], np.r_[0, 1.5], 'k')
    
    rtaxes[0, 0].set_title('all trials')
    rtaxes[0, 1].set_title('easy trials')
    rtaxes[0, 2].set_title('hard trials')
    
    
    #%% check fit to differences in median RT and accuracy across difficulties
    #   and oblique / cardinal criteria
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
    
    is_cardinal = lambda deg: deg == 0 | deg == 90
    
    for ax, cond, label in zip(axes, 
                               [data.easy, data.critDir.map(is_cardinal)], 
                               ['easy - hard', 'cardinal - oblique']):
        ax = helpers.plot_diffs(
                choices_post, rts_post, cond, sim.model.correct, data, 
                label=label, ax=ax)
    
    
    #%% compare dt and ndt distributions for most likely posterior parameters
    sim.model.plot_dt_ndt_distributions(modes, np.eye(modes.size) * 1e-10, pars)
    
    
    #%% 
    for name, mode in modes_tr.iteritems():
        setattr(sim.model, name, mode)
        
    sim.model.plot_example_timecourses(np.arange(100))