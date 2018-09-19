#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:48:53 2018

@author: bitzer
"""

#%% imports
import numpy as np

import helpers
from rtmodels import rtmodel
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
def create_simulator(data, pars, stats='hist', exclude_to=False, 
                     ndtdist='lognorm', fix={}):
    # first of these indicates clockwise rotation, second anti-clockwise
    choices = [1, -1]
    
    model = dirdiffmodel(
        {'directions': data.tarDir, 'criteria': data.critDir}, helpers.dt, 
        maxrt=helpers.maxrt + helpers.dt, toresponse=helpers.toresponse, 
        choices=choices, ndtdist=ndtdist, **fix)
    
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