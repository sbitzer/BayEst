#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:48:53 2018

@author: bitzer
"""

#%% imports
import os, glob
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import helpers
from rtmodels import rtmodel
from rotated_directions import identify_model
from rotated_directions import rotated_directions as dirmodel
from rotated_directions import rotated_directions_diff as dirdiffmodel

import pyEPABC.parameters as parameters


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

    def __init__(self, model, pars, easy, percentiles=np.linspace(1, 99, 5),
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
            
        cw = data[:, 0] == self.model.choices[0]
        acw = data[:, 0] == self.model.choices[1]
        
        if np.any(cw):
            cw_perc = np.percentile(data[cw, 1], self.percentiles)
        else:
            cw_perc = np.zeros(self.percentiles.size, float)
        if np.any(acw):
            acw_perc = np.percentile(data[acw, 1], self.percentiles)
        else:
            acw_perc = np.zeros(self.percentiles.size, float)
        
        if self.exclude_to:
            return np.r_[accuracy, cw_perc, acw_perc]
        else:
            return np.r_[accuracy, cw_perc, acw_perc,
                         (data[:, 0] == self.model.toresponse[0]).mean()]
            

class Stats_hist(object):
    """
    Summary statistics for RT-models with easy and hard trials.
    """

    def __init__(self, model, pars, easy, binq=5, rts=None, 
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
        
        if rts is None:
            if np.isscalar(binq):
                bins = np.linspace(0, helpers.maxrt, binq)
            else:
                assert np.all((binq <= helpers.maxrt) & (binq >= 0))
                bins = binq
        else:
            if np.isscalar(binq):
                # gives np.linspace(10, 90, 5) for binq=5
                offset = -50 / binq + 10
                binq = np.linspace(10 - offset, 90 + offset, binq)
            
            assert np.all((binq < 100) & (binq > 0))
            
            # if there are too many time outs so that the timeout-RT is in the
            # last bin, exclude time outs from computation of percentiles
            # this is to keep all bin edges within maxrt
            perc = np.percentile(rts, binq)
            if perc[-1] == helpers.toresponse[1]:
                perc = np.percentile(rts[rts != helpers.toresponse[1]], binq)
            
            bins = np.r_[0, perc, helpers.maxrt]
            
        # prevent numerical errors in assertion statement below
        bins[-1] -= 1e-15
        
        assert bins.ndim == 1 and bins[0] >= 0 and bins[-1] <= model.maxrt
        
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
        
        if self.exclude_to:
            ind = data[:, 0] != helpers.toresponse[0]
            data = data[ind, :]
        
        # define as float so that you get float division below
        N = float(data.shape[0])
        
        # note that this counts time outs with ind=bins.size
        binind = np.digitize(data[:, 1], self.bins)
        
        B = self.B
        if not self.exclude_to:
            B += 1
        
        return np.array(
                [(data[binind == bi, 0] == self.model.choices[0]).sum() / N
                 for bi in range(1, B+1)])
            

class Stats_id(object):
    """
    Summary statistics for RT-models with easy and hard trials.
    """

    def __init__(self, model, pars, easy):
        pass
    
    def calc(self, data):
        return np.array(data)
    
            
#%% function returning simulator and summary stats for a given subject
def create_simulator(sub, pars, use_liks=False, stats='hist', exclude_to=False, 
                     censor_late=True, ndtdist='lognorm', fix={}):
    data = helpers.load_subject(sub, exclude_to=exclude_to, 
                                censor_late=censor_late)
    
    # first of these indicates clockwise rotation, second anti-clockwise
    choices = [1, -1]
    
    # identify the model based on inferred parameters
    modelstr = identify_model(np.r_[pars.names, fix.keys()])
    if modelstr == 'diff':
        if use_liks is None:
            model = dirdiffmodel(
                {'directions': data.tarDir, 'criteria': data.critDir}, 
                helpers.dt, maxrt=helpers.maxrt + helpers.dt, 
                toresponse=helpers.toresponse, choices=choices, 
                ndtdist=ndtdist, **fix)
        else:
            pass
    else:
        if use_liks:
            Trials, trind, channels = helpers.load_subject_eeg(sub)
            directions = -channels % (2 * np.pi)
            
            data = data.reindex(trind)
            
            model = dirmodel(
                    Trials, dt=helpers.dt, directions=directions,
                    criteria=data.critDir / 180. * np.pi, 
                    maxrt=helpers.maxrt + helpers.dt,
                    toresponse=helpers.toresponse, choices=choices, 
                    ndtdist=ndtdist, trial_dirs=data.tarDir / 180. * np.pi,
                    **fix)
        else:
            model = dirmodel(
                data.tarDir / 180. * np.pi, helpers.dt, 
                data.tarDir.unique() / 180. * np.pi, 
                data.critDir / 180. * np.pi, maxrt=helpers.maxrt + helpers.dt,
                toresponse=helpers.toresponse, choices=choices, 
                ndtdist=ndtdist, **fix)
            
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


#%% helper generating posterior predictive samples from a stored result
def generate_posterior_predictive_data(resultpath, subject, stats, 
                                       store_rw='r+'):
    """Generating, reading and writing posterior predicitive data.
    
    store_rw : str
        'r' - read from store, when possible
        'r+' - read from store, when possible, write result to store, when no
               previous result in store
        'w' - write result to store, ignore previous results
        '' - ignore store, neither read, nor write
    """
    hfile = os.path.join(resultpath, 's%02d_%s.h5' % (subject, stats))
    with pd.HDFStore(hfile, 'r') as store:
        pars = create_default_params(store['parameters'].columns)
        
        if store_rw.startswith('r'):
            if '/choices_post' in store.keys() and '/rts_post' in store.keys():
                choices = store.choices_post
                rts = store.rts_post
                store_rw = 'r'
            else:
                psamples = store['parameters']
                if store_rw == 'r+':
                    store_rw = 'w'
                else:
                    store_rw = 'r'
        else:
            psamples = store['parameters']
            
        exclude_to = store['data_info']['exclude_to']
        censor_late = store['data_info']['censor_late']
        ndtdist = store['ndtdist'][0]
        
        fix = store['fix']
    
    data = helpers.load_subject(subject, exclude_to=exclude_to, 
                                    censor_late=censor_late)
    
    sim, stat, data = create_simulator(
            data, pars, stats, exclude_to, ndtdist, fix)
    
    # will fail with NameError when psamples undefined
    try:
        R = psamples.index.get_level_values('round').unique().size
        pars_post = psamples.loc[R]
        
        resp = sim.sim(pars_post)
        
        choices = resp[:, 0].reshape(sim.model.L, pars_post.shape[0], order='F')
        rts = resp[:, 1].reshape(sim.model.L, pars_post.shape[0], order='F')
    except NameError:
        pass
    
    if store_rw == 'w':
        with pd.HDFStore(hfile, 'r+') as store:
            store['choices_post'] = pd.DataFrame(
                    choices, index=data.index, columns=pars_post.index,
                    dtype=int)
            store['rts_post'] = pd.DataFrame(
                    rts, index=data.index, columns=pars_post.index,
                    dtype=float)
    
    return choices, rts, data, sim.model


def add_posterior_predictive_data(resultdir, stats='hist'):
    """Generate posterior predictive data for all subjects in result and store
    in results file.
    """
    subjects = [int(os.path.basename(f)[1:3]) 
                for f in glob(os.path.join(resultdir, '*%s.log' % stats))]
    
    for sub in subjects:
        print('\rProcessing subject %2d ...' % sub)
        
        # generate posterior predictive data
        try:
            choices_post, rts_post, data, model = (
                    generate_posterior_predictive_data(
                            resultdir, sub, stats, 'r+'))
        except IOError:
            warnings.warn("Skipping subject {} due to IOError!".format(sub))


def create_default_params(parnames):
    pars = parameters.parameter_container()
    
    for pname in parnames:
        if pname in ['diffstd', 'critstd', 'dnoisestd', 'cnoisestd', 
                     'dnoisestd', 'dirstd', 'ndtspread', 'noisestd', 
                     'intstd']:
            pars.add_param(pname, 0, 1, parameters.exponential())
        elif pname == 'bound':
            pars.add_param(pname, 0, 1, parameters.gaussprob(width=0.5, 
                                                             shift=0.5))
        elif pname in ['bias', 'lapseprob', 'lapsetoprob']:
            pars.add_param(pname, 0, 1, parameters.gaussprob())
        elif pname in ['cpsqrtkappa']:
            pars.add_param(pname, 0, 1, parameters.zero())
        else:
            pars.add_param(pname, 0, 1)
            
    return pars


def estimate_posterior_fit(resultpath, subjects, stats, 
                           within_deviance=lambda x: np.sum(x ** 2, axis=1), 
                           across_deviance=lambda x: np.sum(np.abs(x), axis=0),
                           compute_easyhard=True, show=False):
    """Estimates goodness of fit across samples from posterior.
    
    The measure used can be defined by the user. Returned will be
    `across_deviance(within_deviance(posterior_sample - observation))` where
    within_deviance aggregates across dimensions of a single observation and
    across_deviance aggregates across samples from the posterior. So 
    within_deviance has to operate on columns and across_deviance operates on
    rows.
    """
    try:
        len(subjects)
    except TypeError:
        subjects = [subjects]
        
    hddmresult = 'hddm' in resultpath
        
    fit = []
    easyhard = []
    for sub in subjects:
        if hddmresult:
            with pd.HDFStore(resultpath, 'r') as store:
                opts = store['scalar_opt']
                exclude_to = opts.exclude_to
                censor_late = opts.censor_late
                
                # dummy variables for creation of stat object
                pars = create_default_params(['bias', 'noisestd'])
                ndtdist = 'uniform'
                fix = pd.Series({'bias': 0})
        else:
            hfile = os.path.join(resultpath, 's%02d_%s.h5' % (sub, stats))
            with pd.HDFStore(hfile, 'r') as store:
                pars = create_default_params(store['parameters'].columns)
                exclude_to = store['data_info']['exclude_to']
                censor_late = store['data_info']['censor_late']
                ndtdist = store['ndtdist'][0]
                fix = store['fix']
                try:
                    R = (store['parameters'].index
                         .get_level_values('round').unique().size)
                
                # this happened when opening a file saved in python2 in python3
                except KeyError:
                    R = (store['parameters'].index
                         .get_level_values(np.bytes_('round')).unique().size)
                
                simdata = store['simdata'].loc[R]
            
        data = helpers.load_subject(sub, exclude_to=exclude_to, 
                                    censor_late=censor_late)

        sim, stat, data = create_simulator(
                data, pars, stats, exclude_to, ndtdist, fix)
        
        obs_xs = stat.calc(data[['response', 'RT']])
        
        if hddmresult:
            with pd.HDFStore(resultpath, 'r') as store:
                simdata = store.select('ppc_data', 'subject=sub')
                
            simdata.sort_index(inplace=True)
                
            # need to set index for easy trials, because the orders of trials
            # differs in the hddm result from the originally loaded data
            stat.easy = simdata.loc[(sub, 0), 'easy'].values
                
            simdata = simdata[['response', 'rt']]
            
            # convert back into -1, 1 coding of response, see infer_HDDM.py
            simdata['response'] = simdata['response'] * 2 - 1
            
            # convert posterior predictive samples to summary statistics 
            # samples, but put response in right format and make sure that
            # 'easy' association of trials is correct
            simdata = pd.DataFrame(np.concatenate(
                    [stat.calc(simdata.loc[(sub, sa)])[None, :] for
                     sa in simdata.index.get_level_values('sample').unique()]))
        
        fit.append(across_deviance(within_deviance(simdata - obs_xs)))
        
        if compute_easyhard:
            N2 = obs_xs.size // 2
            eh = [within_deviance(obs_xs[None, :N2] - obs_xs[None, N2:])]
            eh.append(within_deviance(simdata.values[:, :N2] 
                                      - simdata.values[:, N2:]))
            eh.append(across_deviance(eh[1] - eh[0]))
        else:
            eh = []
        easyhard.append(eh)
        
        if show:
            fig, ax = plt.subplots()
            
            ax.boxplot(simdata.values)
            ax.plot(np.arange(obs_xs.size) + 1, obs_xs, '*', color='C0')
    
    if np.isscalar(fit[0]):
        fit = pd.Series(fit, index=subjects)
    else:
        fit = pd.DataFrame(fit, index=subjects)
    
    return fit, pd.Series(easyhard, index=subjects)