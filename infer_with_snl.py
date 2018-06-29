#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:00:24 2018

@author: bitzer
"""
#%% imports
import numpy as np
import pandas as pd

import helpers
from rtmodels import rtmodel
from rotated_directions import rotated_directions as dirmodel

import snl
import pyEPABC.parameters as parameters
from pyEPABC.parameters import exponential, gaussprob

import matplotlib.pyplot as plt


#%% define the simulator needed in SNL
class Simulator(object):
    """
    Simulator of timed decisions about rotated motion directions.
    """

    def __init__(self, model, pars):

        if isinstance(model, rtmodel):
            self.model = model
        else:
            raise TypeError("Simulator is only defined for rtmodel!")
        
        if isinstance(pars, parameters.parameter_container):
            self.pars = pars
        else:
            raise TypeError(
                    "Simulator is only defined for parameter_container!")
        
        
    def sim(self, ps, rng=None):

        ps = np.asarray(ps, float)

        if ps.ndim == 1:
            ps = ps[None, :]

        assert ps.ndim == 2

        ps = self.pars.transform(ps)

        ch, rt = self.model.gen_response_with_params(
                np.tile(np.arange(self.model.L), ps.shape[0]), 
                np.repeat(ps, self.model.L, axis=0), self.pars.names)

        return np.c_[ch, rt]
    

#%% define the summary statistics
class Stats_quant(object):
    """
    Summary statistics for RT-models with easy and hard trials.
    """

    def __init__(self, model, pars, easy, percentiles=np.linspace(1, 99, 8),
                 exclude_to=True):

        if isinstance(model, rtmodel):
            self.model = model
        else:
            raise TypeError("Simulator is only defined for rtmodel!")
        
        if isinstance(pars, parameters.parameter_container):
            self.pars = pars
        else:
            raise TypeError(
                    "Simulator is only defined for parameter_container!")
        
        if easy.dtype == np.bool and easy.ndim == 1 and easy.size == model.L:
            self.easy = easy
        else:
            raise TypeError(
                    "easy must be a 1D boolean array of length model.L")
            
        self.percentiles = percentiles
        self.exclude_to = exclude_to

    
    def calc(self, data):

        if data is None:
            return None

        data = np.asarray(data)

        assert data.shape[0] % self.model.L == 0

        n_sims = data.shape[0] // self.model.L

        stats = []
        for sim in range(n_sims):
            data1 = data[sim * self.model.L : (sim + 1) * self.model.L, :]
            
            stats.append(
                    np.r_[self.get_summary_stats(data1, self.easy),
                          self.get_summary_stats(data1, ~self.easy)][None, :])
            
        return np.concatenate(stats) if n_sims > 1 else stats[0][0]
    
            
    def get_summary_stats(self, data, cond):
        data = data[cond, :]
        correct = self.model.correct[cond] == data[:, 0]
        
        if self.exclude_to:
            ind = data[:, 0] != 0
            correct = correct[ind]
            data = data[ind, :]
            
        if correct.size == 0:
            accuracy = 0
        else:
            accuracy = correct.mean()
            
        if self.exclude_to:
            return np.r_[accuracy, np.percentile(data[:, 1], self.percentiles)]
        else:
            return np.r_[accuracy, np.percentile(data[:, 1], self.percentiles),
                         (data[:, 0] == self.model.toresponse[0]).mean()]
            

class Stats_hist(object):
    """
    Summary statistics for RT-models with easy and hard trials.
    """

    def __init__(self, model, pars, easy, bins=7, rts=None, 
                 exclude_to=False):

        if isinstance(model, rtmodel):
            self.model = model
        else:
            raise TypeError("Simulator is only defined for rtmodel!")
        
        if isinstance(pars, parameters.parameter_container):
            self.pars = pars
        else:
            raise TypeError(
                    "Simulator is only defined for parameter_container!")
        
        if easy.dtype == np.bool and easy.ndim == 1 and easy.size == model.L:
            self.easy = easy
        else:
            raise TypeError(
                    "easy must be a 1D boolean array of length model.L")
        
        if np.isscalar(bins):
            B = bins // 2 + 1
            
            median = np.median(rts)
            
            bins = np.r_[
                    median - np.exp(np.arange(B, 0, -1) + np.log(median) - B),
                    median + np.exp(np.arange(B) + 1 
                                    + np.log(model.maxrt - median) - B)]
            
            # prevent numerical errors in assertion statement below
            bins[-1] -= 1e-15
        
        assert bins.ndim == 1 and bins[0] >= 0 and bins[-1] <= model.maxrt
        
        if not exclude_to:
            bins = np.r_[bins, self.model.toresponse[1]]
        
        self.exclude_to = exclude_to
        
        self.bins = bins
        self.B = bins.size - 1

    
    def calc(self, data):

        if data is None:
            return None

        data = np.asarray(data)

        assert data.shape[0] % self.model.L == 0

        n_sims = data.shape[0] // self.model.L

        stats = []
        for sim in range(n_sims):
            data1 = data[sim * self.model.L : (sim + 1) * self.model.L, :]
            
            stats.append(
                    np.r_[self.get_summary_stats(data1, self.easy),
                          self.get_summary_stats(data1, ~self.easy)][None, :])
            
        return np.concatenate(stats) if n_sims > 1 else stats[0][0]
    
            
    def get_summary_stats(self, data, cond):
        data = data[cond, :]
        correct = self.model.correct[cond] == data[:, 0]
        
        if self.exclude_to:
            ind = data[:, 0] != 0
            correct = correct[ind]
            data = data[ind, :]
            
        dens, _ = np.histogram(data[:, 1], self.bins, density=True)
        if correct.size == 0:
            accuracy = 0
        else:
            accuracy = correct.mean()
            
        return np.r_[accuracy, dens]
            

class Stats_id(object):
    """
    Summary statistics for RT-models with easy and hard trials.
    """

    def __init__(self, model, pars, easy):
        pass
    
    def calc(self, data):
        return np.array(data)
    
            
#%% function returning simulator and summary stats for a given subject
def create_simulator(data, pars, stats='hist', exclude_to=False, 
                     ndtdist='lognorm'):
    # first of these indicates clockwise rotation, second anti-clockwise
    choices = [1, -1]
    
    model = dirmodel(
        data.tarDir / 180. * np.pi, helpers.dt, 
        data.tarDir.unique() / 180. * np.pi, data.critDir / 180. * np.pi, 
        maxrt=helpers.maxrt + helpers.dt, toresponse=helpers.toresponse, 
        choices=choices, ndtdist=ndtdist)
    
    if stats == 'hist':
        stats = Stats_hist(model, pars, data['easy'], rts=data.RT, 
                           exclude_to=exclude_to)
    elif stats == 'quant':
        stats = Stats_quant(model, pars, data['easy'], exclude_to=exclude_to)
    elif stats == 'id':
        stats = Stats_id(model, pars, data['easy'])
    else:
        raise ValueError('Unknown type of summary statistics!')
    
    return Simulator(model, pars), stats, data


#%% define parameters and their prior
ndtdist = 'uniform'    

pars = parameters.parameter_container()
pars.add_param('noisestd', 0, 1.2, exponential())
#pars.add_param('intstd', 0, 1.2, exponential())
pars.add_param('bound', 0, 1, gaussprob(width=0.5, shift=0.5))
pars.add_param('bias', 0, .2, gaussprob())
pars.add_param('ndtloc', -2, 1)
pars.add_param('ndtspread', np.log(0.2), 1, exponential())
pars.add_param('lapseprob', -1.65, 1, gaussprob()) # median approx at 0.05
pars.add_param('lapsetoprob', 0, 1, gaussprob())

prior = snl.pdfs.Gaussian(m=pars.mu, S=pars.cov)


#%%
sub = 19
censor_late = True
exclude_to = False
data = helpers.load_subject(sub, exclude_to=exclude_to, censor_late=censor_late)

stats = 'quant'
if stats == 'id':
    conditions = data.easy
else:
    conditions = None

sim, stat, data = create_simulator(data, pars, stats, exclude_to, ndtdist)
p = pars.sample(10)

stat.calc(sim.sim(p))

obs_xs = stat.calc(data[['response', 'RT']])


#%%
fbase = 'test_' + stats
snl.run_snl(obs_xs, sim, stat, prior, filebase=fbase, conditions=conditions)

with pd.HDFStore(fbase + '.h5', 'r+') as store:
    # save extra information about this result
    store['prior_mu'] = pd.Series(pars.mu, pars.names)
    store['prior_cov'] = pd.DataFrame(pars.cov, columns=pars.names, 
                                      index=pars.names)
    store['data_info'] = pd.Series(
            [sub, censor_late, exclude_to], 
            index=['subject', 'censor_late', 'exclude_to'])
    store['stats'] = pd.Series(stats)
    if stats == 'hist':
        store['stats_opt'] = pd.Series(stat.bins)
    elif stats == 'quant':
        store['stats_opt'] = pd.Series(stat.percentiles)
    
    # load stored results
    psamples = store['parameters']
    xsamples = store['simdata']
    logdens = store['logdens']
    trainlog = store['trainlog']
    
    print('elapsed: %s' % store['elapsed'].loc['elapsed'])

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


#%% compare dt and ndt distributions for most likely posterior parameters
sim.model.plot_dt_ndt_distributions(modes, np.eye(modes.size) * 1e-10, pars)


#%% 
for name, mode in modes_tr.iteritems():
    setattr(sim.model, name, mode)
    
sim.model.plot_example_timecourses(np.arange(100))