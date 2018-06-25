#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:00:24 2018

@author: bitzer
"""
#%% imports
from datetime import datetime
import numpy as np

import helpers
from rtmodels import rtmodel
from rotated_directions import rotated_directions as dirmodel

import snl
import pyEPABC.parameters as parameters
from pyEPABC.parameters import exponential, gaussprob


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
class Stats(object):
    """
    Summary statistics for RT-models with easy and hard trials.
    """

    def __init__(self, model, pars, easy, percentiles=np.linspace(5, 95, 7),
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
            return np.r_[0.5, 10.0 * np.ones(7)]
        else:
            return np.r_[correct.mean(), 
                         np.percentile(data[:, 1], self.percentiles)]
            

class Stats_id(object):
    """
    Summary statistics for RT-models with easy and hard trials.
    """

    def __init__(self, model, pars, easy):
        pass
    
    def calc(self, data):
        return np.array(data)
    
            
#%% function returning simulator and summary stats for a given subject
def create_simulator(data, pars, conditions):
    # first of these indicates clockwise rotation, second anti-clockwise
    choices = [1, -1]
    
    model = dirmodel(
        data.tarDir / 180. * np.pi, helpers.dt, 
        data.tarDir.unique() / 180. * np.pi, data.critDir / 180. * np.pi, 
        maxrt=helpers.maxrt + helpers.dt, toresponse=helpers.toresponse, 
        choices=choices)
    
    if conditions is None:
        stats = Stats(model, pars, data['easy'])
    else:
        stats = Stats_id(model, pars, data['easy'])
    
    return Simulator(model, pars), stats, data


#%% define parameters and their prior
pars = parameters.parameter_container()
pars.add_param('noisestd', 0, 1.2, exponential())
pars.add_param('intstd', 0, 1.2, exponential())
pars.add_param('bound', 0, 1, gaussprob(width=0.5, shift=0.5))
pars.add_param('bias', 0, .2, gaussprob())
pars.add_param('ndtmean', -2, 1)
pars.add_param('ndtspread', np.log(0.2), 1, exponential())
pars.add_param('lapseprob', -1.65, 1, gaussprob()) # median approx at 0.05
pars.add_param('lapsetoprob', 0, 1, gaussprob())

prior = snl.pdfs.Gaussian(m=pars.mu, S=pars.cov)


#%%
data = helpers.load_subject(19, exclude_to=False, censor_late=True)

# set to None for RT quantile summary stats, else set to data['easy']
conditions = None

sim, stat, data = create_simulator(data, pars, conditions)
p = pars.sample(10)

stat.calc(sim.sim(p))

obs_xs = stat.calc(data[['response', 'RT']])


#%%
startt = datetime.now()
all_ps, all_xs, lik = snl.run_snl(obs_xs, sim, stat, prior, 
                                  conditions=conditions,
                                  minibatch=100)
endt = datetime.now()

print('elapsed time: ' + (endt - startt).__str__())


#%% 
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.boxplot(all_xs[-1])
ax.plot(np.arange(obs_xs.size) + 1, obs_xs, '*', color='C0')


#%%
import seaborn as sns

ptr = pars.transform(all_ps[-1])

sns.jointplot(ptr[:, 1], ptr[:, 2])


#%% posterior predictive check of complete distribution
# use only median to predict
#pars_post = np.tile(np.median(all_ps[-1], axis=0)[None, :], (1000, 1))
# use full posterior
pars_post = all_ps[-1]

resp = sim.sim(pars_post)

choices_post = resp[:, 0].reshape(sim.model.L, all_ps[-1].shape[0], order='F')
rts_post = resp[:, 1].reshape(sim.model.L, all_ps[-1].shape[0], order='F')

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
