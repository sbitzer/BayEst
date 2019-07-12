#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:06:11 2018

@author: bitzer
"""

from __future__ import print_function, division

import re
import math
import random
import numpy as np
from numba import jit, prange, vectorize, int16, int32, int64, float64
from warnings import warn
from rtmodels import rtmodel
import matplotlib.pyplot as plt


def identify_model(pars):
    """Tries to identify model by checking inferred parameters.
    
    Returns 
        'diff' for rotated_directions_diff 
        'base' for rotated_directions
    """
    
    try:
        parnames = pars.names
    except:
        parnames = np.array(pars)
    
    if np.any(np.isin(
            np.array(['diffstd', 'cpsqrtkappa', 'critstd', 'cnoisestd', 
                      'dnoisestd']), 
            parnames)):
    
        assert np.all(np.isin(
                parnames, rotated_directions_diff.parnames))
    
        return 'diff'
    else:
        assert np.all(np.isin(
                parnames, rotated_directions.parnames))
        
        return 'base'

class rotated_directions(rtmodel):

    @property
    def criteria(self):
        """The criterion orientations to which we compare in each trial.
        
        As the criterion orientations are not directional, they must be within
        [0, pi] only, where 0 is a horizontal criterion and pi/2 a vertical 
        criterion. Given values larger than pi will be mapped to values within
        pi using modulo. Then negative values will be mapped to positive by 
        pi - value.
        """
        return self._criteria
    
    @criteria.setter
    def criteria(self, crit):
        crit = np.array(crit)
        
        if crit.ndim == 0:
            self._criteria = crit * np.ones(self.L)
        elif crit.ndim == 1:
            self._criteria = crit
        else:
            raise ValueError("Could not recognise format of criterion "
                             "orientations! Provide criteria as scalar, if "
                             "all trials use the same criterion, else provide "
                             "a 1D array!")
        
        # this is a neat trick that maps positive and negative angles into 
        # [0, pi] such that absolute values greater than pi are circled back
        # to 0 through the modulus operation while at the same time negative
        # values are mapped to positive ones with pi - value, this works, 
        # because the python modulus operator returns 
        # val - pi * floor(val / pi) = pi - abs(val) % pi
        self._criteria = self._criteria % np.pi
        
        if not self._during_init:
            self.check_balance()
            self.gen_correct()
    
    @property
    def directions(self):
        """The set of motion directions that may be observed.
        
        Directions are defined as angles in [0, 2*pi] where 0 is horizontal 
        motion moving to the right and pi/2 is vertical motion moving to the 
        top. Angles outside of [0, 2*pi] are appropriately mapped into this 
        range adhering to the given definition.
        """
        return self._directions
    
    @directions.setter
    def directions(self, directions):
        if isinstance(directions, np.ndarray):
            if not directions.ndim == 1:
                raise ValueError("directions should be provided in a one-"
                                 "dimensional array.")
            else:
                self._directions = directions
        elif np.isscalar(directions):
            # equally distribute ori# directions from -pi to pi
            directions = int(directions)
            self._directions = (np.linspace(0, 1 - 1/directions, directions) 
                                  * 2 * math.pi - math.pi)
        else:
            raise ValueError("Could not recognise format of given directions"
                             "! Provide them as int for equally spaced "
                             "directions, or as 1D array!")
        
        # map into [0, 2*pi] appropriately handling negative angles, see same
        # operation for criteria above for explanation
        self._directions = self._directions % (2 * np.pi)
        
        if not self._during_init:
            self.check_balance()
            self.gen_correct()
            
    @property
    def D(self):
        """Number of directions considered in model."""
        return self.directions.size
    
    def check_balance(self):
        """Checks whether design is balanced and warns if not.
        
        A design is balanced, when for all criteria there is an equal number
        of directions that are clockwise or anti-clockwise rotated.
        """
        criteria = np.unique(self.criteria)
        
        for crit in criteria:
            cw, acw, ontop, between = get_rotations(self.directions, crit)
            if cw.sum() != acw.sum():
                warn("Unbalanced design: For criterion %4.2f there are %d "
                     "clockwise, but %d anti-clockwise rotations!" % (
                             crit, cw.sum(), acw.sum()))

    @property
    def correct(self):
        """The correct choice for each trial."""
        return self._correct
                
    def gen_correct(self, dirs=None):
        """Tries to figure out correct choice from directions and criteria."""
        
        if self.use_liks and dirs is None:
            correct = np.full(self.L, np.nan)
        else:
            if dirs is not None:
                assert dirs.size == self.L
                dirs = np.array(dirs)
                
            correct = np.zeros(self.L)
            for tr in range(self.L):
                if self.use_features:
                    # determine average direction in trial from stored features
                    direction = np.atan2(np.sin(self._Trials[:, tr]).sum(),
                                         np.cos(self._Trials[:, tr]).sum())
                elif dirs is not None:
                    direction = dirs[tr]
                else:
                    direction = self.directions[self._Trials[tr]]
                
                direction = np.atleast_1d(direction)
                
                cw, acw, ontop, between = get_rotations(direction,
                                                        self.criteria[tr])
                
                if cw[0]:
                    correct[tr] = self.choices[0]
                elif acw[0]:
                    correct[tr] = self.choices[1]
                else:
                    correct[tr] = self.toresponse[0]
            
        self._correct = correct
        
    @property
    def use_features(self):
        """Whether the model uses observed features as input, or just stored 
           directions."""
        return True if self._Trials.ndim == 2 else False
        
    @property
    def use_liks(self):
        """Whether the model uses pre-computed log-likelihoods as input."""
        return True if self._Trials.ndim == 3 else False
    
    @property
    def Trials(self):
        """Trial information used by the model.
        
        Either a 1D, or 2D (use_features=True), or 3D (use_liks=True)
        numpy array.
        
        When 1D, Trials contains the code of the correct choice in that trial.
        When 2D, Trials contains the stream of feature values that the subject
                 may have seen in all the trials of the experiment. Then,
                     S, L = Tials.shape
                 where S is the length of the sequence in each trial
        When 3D, Trials contains pre-computed likelihood values. Then,
                     S, D, L = Tials.shape
                 where D is the number of directions considered in the model
        """
        if self.use_features or self.use_liks:
            return self._Trials
        else:
            return self.directions[self._Trials]
        
    @Trials.setter
    def Trials(self, Trials):
        Trials = np.array(Trials)
        
        # if Trials is log-likelihoods
        if Trials.ndim == 3:
            # don't need to do anymore than just set Trials internally
            self._Trials = Trials
        else:
            # ensure that Trials is within [0, 2*pi]
            Trials = np.array(Trials) % (2 * np.pi)
            
            if Trials.ndim == 2:
                self._Trials = Trials
            elif Trials.ndim == 1:
                # check that Trials only contains valid directions
                if np.all(np.isin(np.unique(Trials), self.directions)):
                    # transform to indices into self.directions
                    self._Trials = np.array([
                            np.flatnonzero(self.directions == i)[0]
                            for i in Trials])
                else:
                    raise ValueError('Trials may only contain valid choice codes' +
                                     ' when features are not used.')
            else:
                raise ValueError('Trials has unknown format, please check!')
        
        self._L = Trials.shape[-1]
        
        if not self._during_init:
            self.gen_correct()
        
    
    @property
    def S(self):
        """Number of time steps maximally simulated by the model."""
        if self.Trials.ndim > 1:
            return self.Trials.shape[0]
        else:
            # the + 1 ensures that time outs can be generated
            return int(math.ceil(self.maxrt / self.dt)) + 1
    
    parnames = ['bound', 'bstretch', 'bshape', 'noisestd', 'intstd', 'prior', 
                'bias', 'ndtloc', 'ndtspread', 'lapseprob', 'lapsetoprob']

    @property
    def prior(self):
        "Prior probabilities over directions."
        return self._prior
        
    @prior.setter
    def prior(self, prior):
        if np.isscalar(prior) and self.D == 2:
            self._prior = np.array([prior])
        elif type(prior) is np.ndarray and prior.size == self.D-1:
            self._prior = prior
        else:
            raise TypeError("The prior should be a numpy array with D-1 "
                            "elements! For two directions only you may also "
                            "provide a scalar.")

    prior_re = re.compile('(?:prior)(?:_(\d))?$')
    
    @property
    def P(self):
        "number of parameters in the model"
        
        # the prior adds D-1 parameters, one of which is counted by its name
        return len(self.parnames) + self.D - 2
    
    
    def __init__(self, Trials, dt=1, directions=8, criteria=0, prior=None, 
                 bias=0.5, noisestd=1, intstd=1, bound=0.8, bstretch=0, 
                 bshape=1.4, ndtloc=-12, ndtspread=0, lapseprob=0.05,
                 lapsetoprob=0.1, ndtdist='lognormal', trial_dirs=None,
                 **rtmodel_args):
        super(rotated_directions, self).__init__(**rtmodel_args)
            
        self._during_init = True
        
        self.name = 'Discrete directions model'
        
        # Time resolution of model simulations.
        self.dt = dt
        
        # set directions that are estimated
        self.directions = directions
        
        # Trial information used by the model.
        self.Trials = Trials
        
        # criterion orientations
        self.criteria = criteria
        
        # Prior probabilities over directions.
        if prior is None:
            self.prior = np.ones(self.D-1) / self.D
        else:
            self.prior = prior
            
        # choice bias as prior probability that clockwise is correct
        self.bias = bias
            
        # Standard deviation of noise added to feature values.
        self.noisestd = noisestd
            
        # Standard deviation of internal uncertainty.
        self.intstd = intstd
            
        # Bound that needs to be reached before decision is made.
        # If collapsing, it's the initial value.
        self.bound = bound
            
        # Extent of collapse for bound, see boundfun.
        self.bstretch = bstretch
        
        # Shape parameter of the collapsing bound, see boundfun
        self.bshape = bshape
            
        # which ndt distribution to use?
        self.ndtdist = ndtdist
        
        # location of nondecision time distribution
        self.ndtloc = ndtloc
            
        # Spread of nondecision time.
        self.ndtspread = ndtspread
            
        # Probability of a lapse.
        self.lapseprob = lapseprob
            
        # Probability that a lapse will be timed out.
        self.lapsetoprob = lapsetoprob
        
        # figure out the correct choice in each trial
        self.gen_correct(trial_dirs)
        
        # check whether design is balanced (and warn if not)
        self.check_balance()
        
        self._during_init = False
        

    def __str__(self):
        info = super(rotated_directions, self).__str__()
        
        # empty line
        info += '\n'
        
        # model-specific parameters
        info += 'directions:\n'
        info += self.directions.__str__() + '\n'
        info += 'uses features: %4d' % self.use_features + '\n'
        info += 'dt           : %8.3f' % self.dt + '\n'
        info += 'bound        : %8.3f' % self.bound + '\n'
        info += 'bstretch     : %7.2f' % self.bstretch + '\n'
        info += 'bshape       : %7.2f' % self.bshape + '\n'
        info += 'noisestd     : %6.1f' % self.noisestd + '\n'
        info += 'intstd       : %6.1f' % self.intstd + '\n'
        info += 'ndtdist      : %s'    % self.ndtdist + '\n'
        info += 'ndtloc       : %7.2f' % self.ndtloc + '\n'
        info += 'ndtspread    : %7.2f' % self.ndtspread + '\n'
        info += 'lapseprob    : %7.2f' % self.lapseprob + '\n'
        info += 'lapsetoprob  : %7.2f' % self.lapsetoprob + '\n'
        info += 'bias         : %7.2f' % self.bias + '\n'
        info += 'prior        : ' + ', '.join(map(lambda s: '{:8.3f}'.format(s), 
                                                  self.prior)) + '\n'
        
        return info


    def estimate_memory_for_gen_response(self, N):
        """Estimate how much memory you would need to produce the desired responses."""
        
        mbpernum = 8 / 1024 / 1024
        
        # (for input features + for input params + for output responses)
        return mbpernum * N * (self.S + self.P + 2)
    
    
    def gen_timecourses(self, trind):
        """Generate example time courses of internal decision variables.
        
        Returned:
            times - simulated time points (starting from 0 for prior)
            logprob_cw - posterior log-probability of clockwise rotation after
                         renormalisation so that clockwise and anit-clockwise
                         directions sum to 1
            logprob - posterior log-probability of all considered motion 
                      directions
            logpost - unnormalised log-posterior values of all directions
            logliks - unnormalised log-likelihood values of all directions
            
            1st dim: time points, 2nd dim: directions, 3rd dim: trials
            for logprob_cw: 2nd dim: trials
        """
        N = trind.size
        
        times = np.arange(0, self.S) * self.dt
        
        logprior = np.r_[np.log(self.prior), np.log(1 - self.prior.sum())]
        
        logliks = np.zeros((self.S-1, self.D, N))
        logpost = np.tile(logprior[None, :, None], (self.S, 1, N))
        logprob = np.zeros_like(logpost)
        logprob_cw = np.zeros((self.S, N))
        for i, tr in enumerate(trind):
            cw, acw, ontop, between = get_rotations(self.directions, 
                                                    self.criteria[tr])
            
            if self.use_liks:
                logliks[:, :, i] = self.Trials[:-1, :, i] / self.intstd ** 2
            else:
                # sample feature values (observed directions)
                if self.use_features:
                    features = self.Trials[:, tr]
                else:
                    features = np.full(self.S-1, self.directions[self._Trials[tr]])
                # add von Mises noise
                features = (
                        np.random.vonmises(0, 1 / self.noisestd ** 2, self.S-1)
                        + features) % (2 * np.pi)
                
                # unnormalised log-likelihood values
                logliks[:, :, i] = (
                        np.cos(features[:, None] - self.directions[None, :])
                        / self.intstd ** 2)
            
            # unnormalised log-posterior values
            logpost[1:, :, i] = self.dt * logliks[:, :, i]
            logpost[:, :, i] = np.cumsum(logpost[:, :, i], axis=0)
        
            # posterior log-probabilities
            logprob[:, :, i] = (logpost[:, :, i] 
                                - logsumexp_2d(logpost[:, :, i], axis=1))
            
            lp_cw = (logsumexp_2d(logprob[:, cw, i], axis=1) 
                     + np.log(self.bias))
            lp_acw = (logsumexp_2d(logprob[:, acw, i], axis=1)
                      + np.log(1 - self.bias))
            
            logprob_cw[:, i] = (
                    lp_cw - logsumexp_2d(np.c_[lp_cw, lp_acw], axis=1))[:, 0]
            
        return times, logprob_cw, logprob, logpost, logliks
    
    
    def plot_example_timecourses(self, trind, dvtype='prob', with_bound=True,
                                 seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        times, logprob_cw, logprob, logpost, logliks = self.gen_timecourses(
                trind)
        
        labels = {self.toresponse[0]: 'ambiguous', 
                  self.choices[0]: 'clockwise', 
                  self.choices[1]: 'anti-clockwise'}
        cols = {self.toresponse[0]: 'k', 
                self.choices[0]: 'C0', 
                self.choices[1]: 'C1'}
        
        if dvtype == 'prob':
            dv = np.exp(logprob_cw)
        elif dvtype == 'logprob':
            dv = logprob_cw
        elif dvtype == 'logprobdiff':
            dv = logprob_cw - np.log(1 - np.exp(logprob_cw))
        
        fig, ax = plt.subplots()
        for c in np.r_[self.toresponse[0], self.choices]:
            ind = self.correct[trind] == c
            if ind.sum() > 0:
                lines = ax.plot(times, dv[:, ind], 
                                color=cols[c], alpha=.3)
                lines[0].set_label(labels[c])
        
        if with_bound:
            lower = 1 - self.bound
            upper = self.bound
            if dvtype == 'logprob':
                lower = np.log(lower)
                upper = np.log(upper)
            elif dvtype == 'logprobdiff':
                upper = np.log(upper / (1 - upper))
                lower = -upper
                
            line, = ax.plot(times, np.ones_like(times) * upper, 'k')
            line, = ax.plot(times, np.ones_like(times) * lower, 'k')
            line.set_label('bounds')
        
        ax.legend(loc='upper left')
        ax.set_xlabel('time (s)')
        if dvtype == 'prob':
            ax.set_ylabel('probability clockwise rotation')
        elif dvtype == 'logprob':
            ax.set_ylabel('log-probability clockwise rotation')
        elif dvtype == 'logprobdiff':
            ax.set_ylabel('log p(clockwise) - log p(anti-clockwise)')
        
        return fig, ax
    
    
    def plot_dt_ndt_distributions(self, mean=None, cov=None, pars=None, R=30):
        """Plot decision and non-decision time distributions separately.
        
        Uses either the parameters stored in the model, or the given parameter
        posterior. Decision time distribution will include lapses.
        """
        
        fig, ax = plt.subplots()
        
        if mean is not None:
            params = pars.sample_transformed(self.L * R, mean, cov)
            
            inds = [pars.names[pars.names == 'ndtloc'].index.values[0],
                    pars.names[pars.names == 'ndtspread'].index.values[0]]
        
        # sample from decision time distribution
        if mean is None:
            ndtloc = self.ndtloc
            ndtspread = self.ndtspread
            self.ndtloc = -30
            self.ndtspread = 0
            choices, rts = self.gen_response(np.arange(self.L), rep=R)
            self.ndtloc = ndtloc
            self.ndtspread = ndtspread
        else:
            ndtloc = params[:, inds[0]].copy()
            ndtspread = params[:, inds[1]].copy()
            params[:, inds[0]] = -30
            params[:, inds[1]] = 0
            choices, rts = self.gen_response_with_params(
                    np.tile(np.arange(self.L), R), params, pars.names)
            params[:, inds[0]] = ndtloc
            params[:, inds[1]] = ndtspread
            
        self.plot_response_distribution(choices, rts, ax=ax)
        
        # sample from non-decision time distribution
        if mean is None:
            if self.ndtdist == 'lognormal':
                rts = np.random.lognormal(self.ndtloc, self.ndtspread, 
                                          self.L * R)
            elif self.ndtdist == 'uniform':
                rts = (np.random.rand(self.L * R) * self.ndtspread
                       + np.exp(self.ndtloc))
        else:
            if self.ndtdist == 'lognormal':
                rts = np.array([np.random.lognormal(mu, sig) 
                                for (mu, sig) in params[:, inds]])
            elif self.ndtdist == 'uniform':
                rts = (np.random.rand(params.shape[0]) * params[:, inds[1]]
                       + np.exp(params[:, inds[0]]))
            
        choices = self.choices[np.zeros(self.L * R, dtype=int)]
        
        self.plot_response_distribution(choices, rts, alpha=0.5, ax=ax)
        
        ax.set_xlabel('time (s)')
        ax.set_ylabel('density')
        
        ax.legend([c[0] for c in ax.containers], 
                  ['dt for choice %d' % self.choices[0], 
                   'dt for choice %d' % self.choices[1], 
                   'ndt'])
        
        return fig, ax
    
    
    def gen_response(self, trind, rep=1):
        N = trind.size
        if rep > 1:
            trind = np.tile(trind, rep)
        
        choices, rts = self.gen_response_with_params(trind)
        
        if rep > 1:
            choices = choices.reshape((rep, N))
            rts = rts.reshape((rep, N))
        
        return choices, rts
        
        
    def gen_response_with_params(self, trind, params={}, parnames=None, 
                                 user_code=True):
        if parnames is None:
            assert( type(params) is dict )
            pardict = params
            if 'prior' in pardict:
                if np.isscalar(pardict['prior']):
                    assert self.D == 2, ('prior can only be scalar, if there '
                                         'are only 2 directions')
                    pardict['prior'] = np.array([pardict['prior']])
        else:
            assert( type(params) is np.ndarray )
            pardict = {}
            new_prior = True
            for ind, name in enumerate(parnames):
                match = self.prior_re.match(name)
                if match is None:
                    pardict[name] = params[:, ind]
                else:
                    ind_prior = match.groups()[0]
                    if ind_prior is None:
                        pardict['prior'] = params[:, ind]
                        pardict['prior'] = pardict['prior'][:, None]
                    else:
                        if new_prior:
                            pardict['prior'] = np.full((params.shape[0], 
                                                        self.D-1), np.nan)
                            new_prior = False
                        pardict['prior'][:, int(ind_prior)] = params[:, ind]
        parnames = pardict.keys()
        
        # check whether any of the bound parameters are given
        if any(x in ['bound', 'bstretch', 'bshape'] for x in parnames):
            changing_bound = True
        else:
            changing_bound = False
        
        # get the number of different parameter sets, P, and check whether the 
        # given parameter counts are consistent (all have the same P)
        P = None
        for name in parnames:
            if not np.isscalar(pardict[name]):
                if P is None:
                    P = pardict[name].shape[0]
                else:
                    if P != pardict[name].shape[0]:
                        raise ValueError('The given parameter dictionary ' +
                            'contains inconsistent parameter counts')
        if P is None:
            P = 1
        
        # get the number of trials, N, and check whether it is consistent with 
        # the number of parameters P
        if np.isscalar(trind):
            trind = np.full(P, trind, dtype=int)
        N = trind.shape[0]
        if P > 1 and N > 1 and N != P:
            raise ValueError('The number of trials in trind and the ' +
                             'number of parameters in params does not ' + 
                             'fit together')
        
        NP = max(N, P)
        
        # if continuing would exceed the memory limit
        if self.estimate_memory_for_gen_response(NP) > self.memlim:
            # divide the job in smaller batches and run those
        
            # determine batch size for given memory limit
            NB = int(math.floor(NP / self.estimate_memory_for_gen_response(NP) 
                                * self.memlim))
            
            choices = np.zeros(NP, dtype=np.int8)
            rts = np.zeros(NP)
            
            remaining = NP
            firstind = 0
            while remaining > 0:
                index = np.arange(firstind, firstind + min(remaining, NB))
                if P > 1 and N > 1:
                    trind_batch = trind[index]
                    params_batch = extract_param_batch(pardict, index)
                elif N == 1:
                    trind_batch = trind
                    params_batch = extract_param_batch(pardict, index)
                elif P == 1:
                    trind_batch = trind[index]
                    params_batch = pardict
                else:
                    raise RuntimeError("N and P are not consistent.")
                    
                choices[index], rts[index] = self.gen_response_with_params(
                    trind_batch, params_batch, user_code=user_code)
                
                remaining -= NB
                firstind += NB
        else:
            # make a complete parameter dictionary with all parameters
            # this is quite a bit of waste of memory and should probably be recoded
            # more sensibly in the future, but for now it makes the jitted function
            # simple
            allpars = {}
            for name in self.parnames:
                if name in parnames:
                    allpars[name] = pardict[name]
                else:
                    allpars[name] = getattr(self, name)
                    
                if name == 'prior':
                    if allpars[name].ndim == 1:
                        allpars[name] = np.tile(allpars[name], (N,1))
                    elif allpars[name].shape[0] == 1 and N > 1:
                        allpars[name] = np.tile(allpars[name], (N,1))
                elif np.isscalar(allpars[name]) and N >= 1:
                    allpars[name] = np.full(N, allpars[name], dtype=float)
                elif allpars[name].shape[0] == 1 and N > 1:
                    allpars[name] = np.full(N, allpars[name], dtype=float)
            
            # select input features
            if self.use_liks:
                features = self.Trials
            else:
                if self.use_features:
                    features = self.Trials[:, trind]
                else:
                    features = self.directions[self._Trials[trind]]
                    features = np.tile(features, (self.S, 1))
                
                # expand to 3D so that it's 3D in all cases, because numba
                # doesn't work with varying dimensions
                features = features[:, None, :]
                
            # call the compiled function
            choices, rts = self.gen_response_jitted(
                    features, allpars, self.criteria[trind], changing_bound,
                    trind)
                
            # transform choices to those expected by user, if necessary
            if user_code:
                toresponse_intern = np.r_[-1, self.toresponse[1]]
                timed_out = choices == toresponse_intern[0]
                choices[timed_out] = self.toresponse[0]
                in_time = np.logical_not(timed_out)
                choices[in_time] = self.choices[choices[in_time]]
            
        return choices, rts
        
        
    def gen_response_jitted(self, features, allpars, criteria, changing_bound,
                            trind):
        toresponse_intern = np.r_[-1, self.toresponse[1]]
            
        # call the compiled function
        choices, rts = gen_response_jitted_dir(
                features, self.maxrt, toresponse_intern, 
                self.choices, self.dt, self.directions, criteria, 
                allpars['prior'], allpars['bias'], allpars['noisestd'], 
                allpars['intstd'], allpars['bound'], allpars['bstretch'], 
                allpars['bshape'], allpars['ndtloc'], allpars['ndtspread'], 
                allpars['lapseprob'], allpars['lapsetoprob'], changing_bound,
                0 if self.ndtdist == 'lognormal' else 1, trind)
            
        return choices, rts

@jit(nopython=True, parallel=True)
def gen_response_jitted_dir(
        features, maxrt, toresponse, choices, dt, directions, criteria,
        prior, bias, noisestd, intstd, bound, bstretch, bshape, ndtloc, 
        ndtspread, lapseprob, lapsetoprob, changing_bound, ndtdist, trind):
    
    D = len(directions)
    C = len(choices)
    
    S, D2, N = features.shape
    
    if D2 == D:
        use_liks = True
        N = trind.size
    else:
        use_liks = False
    
    choices_out = np.full(N, toresponse[0], dtype=np.int8)
    rts = np.full(N, toresponse[1], np.float64)
    
    # pre-compute collapsing bound
    boundvals = np.full(S, math.log(bound[0]))
    if bstretch[0] > 0:
        for t in range(S):
            boundvals[t] = math.log( boundfun((t+1.0) / maxrt, bound[0], 
                bstretch[0], bshape[0]) )
    
    for tr in prange(N):
        # is it a lapse trial?
        if random.random() < lapseprob[tr]:
            # is it a timed-out lapse trial?
            if random.random() < lapsetoprob[tr]:
                choices_out[tr] = toresponse[0]
                rts[tr] = toresponse[1]
            else:
                choices_out[tr] = random.randint(0, C-1)
                rts[tr] = random.random() * maxrt
        else:
            # determine directions corresponding to clockwise rotation
            cw, acw, ontop, between = get_rotations(directions, criteria[tr])
            
            logpost = np.zeros(D)
            logpost[:D-1] = np.log(prior[tr, :])
            logpost[-1] = math.log(1 - prior[tr, :].sum())
            
            # for all presented features
            for t in range(S):
                # get current bound value
                if changing_bound:
                    # need to compute boundval from parameters in this trial
                    if bstretch[tr] == 0:
                        boundval = math.log(bound[tr])
                    else:
                        boundval = math.log( boundfun((t+1.0) / maxrt, 
                            bound[tr], bstretch[tr], bshape[tr]) )
                else:
                    # can use pre-computed bound value
                    boundval = boundvals[t]
                
                if not use_liks:
                    # add noise to feature
                    noisy_feature = random.vonmisesvariate(features[t, 0, tr],
                            1 / noisestd[tr]**2)
                
                # compute log-probabilities of directions
                for d in range(D):
                    if use_liks:
                        logpost[d] += (dt * features[t, d, trind[tr]]
                                       / intstd[tr] ** 2)
                    else:
                        logpost[d] += dt * (
                                math.cos(noisy_feature - directions[d])
                                / intstd[tr]**2)
                        
                # normalise
                logprobs = logpost - logsumexp(logpost)
                
                # log-probability of clockwise and anti-clockwise rotation
                # (renormalise to ignore probability mass at directions without
                # information about rotation)
                lp_rot = np.zeros(2)
                lp_rot[0] = logsumexp(logprobs[cw]) + math.log(bias[tr])
                lp_rot[1] = logsumexp(logprobs[acw]) + math.log(1 - bias[tr])
                lp_rot -= logsumexp(lp_rot)
                
                if lp_rot[0] >= boundval or lp_rot[1] >= boundval:
                    if ndtdist == 0:
                        ndt = random.lognormvariate(ndtloc[tr], ndtspread[tr])
                    else:
                        ndt = (random.random() * ndtspread[tr]
                               + math.exp(ndtloc[tr]))
                        
                    # add 1 to t because t starts from 0
                    rts[tr] = (t+1) * dt + ndt
                    
                    if lp_rot[0] >= boundval:
                        choices_out[tr] = 0
                    else:
                        choices_out[tr] = 1
                        
                    break
                
            if rts[tr] > maxrt:
                choices_out[tr] = toresponse[0]
                rts[tr] = toresponse[1]
    
    return choices_out, rts


class rotated_directions_diff(rtmodel):
    """Making decisions based on a difference between estimates of a criterion
       direction and a motion direction."""
    
    @property
    def Trials(self):
        """Trial information used by the model.
        
        Dictionary with keys 'directions' and 'criteria'. Criteria are always
        in a 1D numpy array and give the criterion presented in each trial.
        Directions are either a 1D (use_liks=False), 3D (use_liks=True), or
        4D (use_liks=True) numpy array. In either case it gives information 
        about the motion directions presented in each trial.
        
        When 1D, directions directly contains the presented motion directions  
                 in degrees for each trial
        When 3D, directions contains externally estimated, unnormalised (log-)
                 likelihood values for each considered motion direction for 
                 each time point within the trial. Then,
                     D, S, L = Tials['directions'].shape
                 where D is the number of motion directions encoded in the 
                 model and S is the length of the time sequence in each trial.
                 This will then internally be transformed into the 4D version
                 to comply with the encoding of the model.
        When 4D, directions equally contains externally estimated, unnormalised 
                 likelihood values for each considered motion direction for 
                 each time point within the trial, but they are organised with
                 respect to direction differences and criteria encoded in the
                 model such that
                     DF, 2 * CR, S, L = Trials['directions'].shape
                 where DF is the number of differences and CR the number of
                 criteria encoded in the model and the different motion 
                 directions are encoded as direction = criterion - difference
                 Note that likelihoods need to be provided for upwards and
                 downwards pointing criteria, i.e., Trials['directions'][:, i]
                 i in [0, CR-1] corresponds to a criterion with up direction 
                 and Trials['directions'][:, CR + i] is the corresponding 
                 criterion in down direction. This is necessary, because these
                 likelihoods correspond to the two possible motion directions 
                 that a single criterion, which can either be interpreted as up
                 or down motion, allows.
                 
        Either directions or criteria can also be a scalar, then this scalar
        will be expanded into a 1D array with number of trials determined from
        the other
        """
        return self._Trials
    
    @Trials.setter
    def Trials(self, Trials):
        crit = np.array(Trials['criteria'])
        dirs = np.array(Trials['directions'])
        
        if crit.ndim == 0:
            Lc = 0
        elif crit.ndim == 1:
            Lc = crit.size
        else:
            raise ValueError('criteria in Trials has wrong format (only '
                             'scalar or 1D array allowed)!')
        
        if dirs.ndim == 0:
            Ld = 0
        elif dirs.ndim == 1:
            Ld = dirs.size
        elif dirs.ndim == 3:
            D, S, Ld = dirs.shape
        elif dirs.ndim == 4:
            DF, CR2, S, Ld = dirs.shape
        else:
            raise ValueError('directions in Trials has wrong format (only '
                             'scalar, 1D, or 3D array allowed)!')
        
        # expand scalars
        if Ld == Lc == 0:
            raise ValueError('At least one of criteria or directions needs '
                             'to contain elements for each trial, but I found '
                             'that both are scalars.')
        elif Ld == 0:
            dirs = np.full(Lc, dirs, dtype=np.int16)
            Ld = Lc
        elif Lc == 0:
            crit = np.full(Ld, crit, dtype=np.int16)
            Lc = Ld
            
        if Ld == Lc:
            self._L = Lc
        else:
            raise ValueError('Number of trials given in criteria and '
                             'directions is not equal, but needs to be!')
            
        # ensure that directions and criteria are encoded correctly
        
        # this is a neat trick that maps positive and negative angles into 
        # [0, 180] such that absolute values greater than 180 are circled back
        # to 0 through the modulus operation while at the same time negative
        # values are mapped to positive ones with 180 - value, this works, 
        # because the python modulus operator returns 
        # val - 180 * floor(val / 180) = 180 - abs(val) % 180
        crit = crit.astype(np.int16) % 180
        if self.criteria is None:
            self.criteria = np.unique(crit)
        else:
            # check that criteria only contains valid criteria
            if not np.all(np.isin(np.unique(crit), self.criteria)):
                raise ValueError(
                        'Trials contains criteria that are not ' + 
                        'registered in the model, please check!')
        
        if dirs.ndim == 1:
            dirs = dirs.astype(np.int16) % 360
            if self.directions is None:
                self.directions = np.unique(dirs)
            else:
                # check that Trials only contains valid directions
                if not np.all(np.isin(np.unique(dirs), self.directions)):
                    raise ValueError(
                            'Trials contains directions that are not ' + 
                            'registered in the model, please check!')
        elif dirs.ndim == 3:
            dirs = dirs.astype(float)
            
            if self.directions is None:
                raise ValueError(
                        'It appears that you have created a model which uses '
                        'likelihood values as Trial input, but you have '
                        'forgotten to specify the motion directions '
                        'corresponding to the likelihoods. Please provide '
                        '"directions" in the constructor of the model.')
            
            try:
                self.differences
            except AttributeError:
                # gen_differences uses use_liks so I need to make sure that it
                # can identify the use_lik state from the stored Trials
                self._Trials = {'criteria': crit, 'directions': dirs}
                
                # determine the minimal set of differences needed to represent
                # the directions given the available criterion values
                self.gen_differences()
            
            # determine the directions corresponding to each combination of
            # difference and criterion now encoded in the model
            cr = np.tile(np.r_[self.criteria, 180 + self.criteria], 
                         (self.DF, 1))
            df = np.tile(self.differences[:, None], (1, 2 * self.CR))
            di = np.reshape((cr.flatten() - df.flatten()) % 360, 
                            (self.DF, 2 * self.CR))
            
            # if all directions now defined by criteria and differences are
            # available in the provided set of directions
            if set(np.unique(di)).issubset(set(self.directions)):
                # just figure out the corresponding indices into dirs
                dirinds = np.searchsorted(
                        self.directions, di, 
                        sorter=np.argsort(self.directions))
            else:
                # find indices of those provided directions that are closest
                # to the ones required by criteria and differences
                dirinds = np.zeros_like(di)
                for i in range(self.DF):
                    for j in range(2 * self.CR):
                        dirinds[i, j] = np.argmin(
                                np.abs(self.directions - di[i, j]))
                
                warn("The way the model is now setup the combinations of "
                     "criteria and differences define some directions for "
                     "which no likelihoods are provided in "
                     "Trials['directions']. Will choose likelihood values "
                     "associated with the closest provided directions for "
                     "these directions, but you might want to check the "
                     "setup of the model!")
            
            # copy likelihood values in difference x criterion array
            liks = np.zeros(self.DF, 2 * self.CR, S, Lc)
            for i in range(self.DF):
                for j in range(2 * self.CR):
                    liks[i, j, :, :] = dirs[dirinds[i, j], :, :]
            dirs = liks
        
        elif dirs.ndim == 4:
            dirs = dirs.astype(float)
            
            try:
                self.DF
            except AttributeError:
                if self.directions is None:
                    raise ValueError(
                            'It appears that you have created a model which uses '
                            'likelihood values as Trial input, but you have '
                            'forgotten to specify the motion directions '
                            'corresponding to the likelihoods. Please provide '
                            '"directions" in the constructor of the model.')
                else:
                    # gen_differences uses use_liks so I need to make sure that it
                    # can identify the use_lik state from the stored Trials
                    self._Trials = {'criteria': crit, 'directions': dirs}
                    
                    # determine the minimal set of differences needed to represent
                    # the directions given the available criterion values
                    self.gen_differences()
                    
            if self.DF != DF:
                raise ValueError(
                        "The number of differences used by the model as "
                        "computed from the provided criteria and directions "
                        "does not correspond to the number of differences "
                        "used by Trials['directions']!")
            if self.CR * 2 != CR2:
                raise ValueError(
                        "The number of criteria encoded in the model is not "
                        "compatible with the number of criteria used by "
                        "Trials['directions']!")
        
        self._Trials = {'criteria': crit, 'directions': dirs}
        
        self.gen_correct()
        self.check_balance()
            
    @property
    def use_liks(self):
        """Whether the model uses externally estimated likelihoods as input, 
            or just stored directions."""
        if self.Trials['directions'].ndim == 1:
            return False
        else:
            return True
    
    @property
    def directions(self):
        """The set of motion directions that may be observed.
        
        Directions are defined as angles in [0, 360) where 0 is horizontal 
        motion moving to the right and 90 is vertical motion moving up.
        Angles outside of [0, 360) are appropriately mapped into this 
        range adhering to the given definition.
        """
        return self._directions
    
    @directions.setter
    def directions(self, directions):
        if directions is None:
            self._directions = None
        else:
            if isinstance(directions, np.ndarray):
                if not directions.ndim == 1:
                    raise ValueError("directions should be provided in a one-"
                                     "dimensional array.")
                else:
                    self._directions = directions
            elif np.isscalar(directions):
                # equally distribute ori# directions from -pi to pi
                directions = int(directions)
                self._directions = (np.linspace(0, 1 - 1/directions, directions) 
                                    * 360 - 180) % 360
            else:
                raise ValueError("Could not recognise format of given directions"
                                 "! Provide them as int for equally spaced "
                                 "directions, or as 1D array!")
            
            # map into [0, 360) appropriately handling negative angles, see same
            # operation for criteria above for explanation
            self._directions = (self._directions % 360).astype(np.int16)
            
            if not self._during_init:
                self.gen_differences()
                self.check_balance()
                self.gen_correct()
    
    
    @property
    def D(self):
        """Number of directions considered in model."""
        return self.directions.size
        
    
    @property
    def criteria(self):
        """The criterion orientations encoded in the model.
        
        As the criterion orientations are not directional, they must be within
        [0, 180) only, where 0 is a horizontal criterion and 90 a vertical 
        criterion. Given values larger than 180 will be mapped to values within
        180 using modulo. Then negative values will be mapped to positive by 
        180 - value.
        """
        return self._criteria
    
    @criteria.setter
    def criteria(self, crit):
        if crit is None:
            self._criteria = None
        else:
            crit = np.array(crit)
            
            if crit.ndim == 1:
                self._criteria = np.unique(crit.astype(np.int16))
            else:
                raise ValueError("Could not recognise format of criterion "
                                 "orientations! Please provide criteria as a 1D "
                                 "array!")
            
            # this is a neat trick that maps positive and negative angles into 
            # [0, pi] such that absolute values greater than pi are circled back
            # to 0 through the modulus operation while at the same time negative
            # values are mapped to positive ones with pi - value, this works, 
            # because the python modulus operator returns 
            # val - pi * floor(val / pi) = pi - abs(val) % pi
            self._criteria = self._criteria % 180
            
            if not self._during_init:
                self.gen_differences()
                self.check_balance()
                self.gen_correct()
            
    @property
    def CR(self):
        """Number of criteria encoded in model."""
        return self.criteria.size
    
    
    @property
    def differences(self):
        """Differences encoded in the model.
        
        Computed from given criteria and Trials, or criteria and directions.
        """
        return self._differences
    
    def gen_differences(self):
        """Generates the minimal set of differences needed to encode the 
           directions stored in the model."""
           
        if self.use_liks:
            # all posssible combinations of criteria and directions
            dirs = np.repeat(self.directions, self.CR)
            crit = np.tile(self.criteria, self.D)
        else:
            # only combinations of criteria and direction that occurred in the
            # experiment
            dirs = self.Trials['directions']
            crit = self.Trials['criteria']
        
        self._differences = np.unique(compute_minimal_differences(dirs, crit))
    
    @property
    def DF(self):
        """Number of differences considered in model."""
        return self.differences.size
    
    
    def check_balance(self):
        """Checks whether design is balanced and warns if not.
        
        A design is balanced, when for all criteria there is an equal number
        of directions that are clockwise or anti-clockwise rotated.
        """
        if self.use_liks:
            pass
        else:
            criteria = self.criteria
            
            diffs = compute_minimal_differences(self.Trials['directions'],
                                                self.Trials['criteria'])
            
            for crit in criteria:
                critind = self.Trials['criteria'] == crit
                rotations = np.sign(diffs[critind])
                if rotations.sum() != 0:
                    warn("Unbalanced design: For criterion %3d there are %d "
                         "clockwise, but %d anti-clockwise rotations!" % (
                                 crit, (rotations == 1), (rotations == -1)))
    
    
    @property
    def correct(self):
        """The correct choice for each trial."""
        return self._correct
                
    def gen_correct(self):
        """Figures out correct choice from directions and criteria."""
        
        if self.use_liks:
            self._correct = np.full(self.L, np.nan)
        else:
            # positive diffs correspond to clockwise rotation, negative to 
            # anti-clockwise rotation
            diffs = compute_minimal_differences(self.Trials['directions'], 
                                                self.Trials['criteria'])
            
            correct = np.full_like(diffs, self.choices[0], 
                                   dtype=self.choices.dtype)
            correct[diffs < 0] = self.choices[1]
            
            # difference of 0 is undecidable, encode as time out response
            correct[diffs == 0] = self.toresponse[0]
            
        self._correct = correct
    
    
    @property
    def S(self):
        """Number of time steps maximally simulated by the model."""
        if self.Trials['directions'].ndim == 3:
            return self.Trials['directions'].shape[1]
        else:
            # the + 1 ensures that time outs can be generated
            return int(math.ceil(self.maxrt / self.dt)) + 1
    
    
    parnames = ['diffstd', 'cpsqrtkappa', 'critstd', 'dirstd', 'cnoisestd',
                'dnoisestd', 'bound', 'bstretch', 'bshape', 
                'bias', 'ndtloc', 'ndtspread', 'lapseprob', 'lapsetoprob']
    
    
    @property
    def P(self):
        "number of parameters in the model"
        
        return len(self.parnames)
    
    
    def __init__(self, Trials, dt=1, directions=None, criteria=None, bias=0.5, 
                 diffstd=1, cpsqrtkappa=1, critstd=1, dirstd=1, cnoisestd=1, 
                 dnoisestd=1, bound=0.8, bstretch=0, bshape=1.4, 
                 ndtloc=-12, ndtspread=0, lapseprob=0.05, lapsetoprob=0.1, 
                 ndtdist='lognormal', **rtmodel_args):
        super(rotated_directions_diff, self).__init__(**rtmodel_args)
        
        self._during_init = True
        
        self.name = 'Discrete direction difference model'
        
        # Time resolution of model simulations.
        self.dt = dt
        
        # set directions that are estimated
        self.directions = directions
        
        # criterion orientations
        self.criteria = criteria
        
        # Trial information used by the model.
        self.Trials = Trials
        
        # generate the differences from the given information
        self.gen_differences()
        
        # choice bias as prior probability that clockwise is correct
        self.bias = bias
        
        # determines prior over difference magnitudes 
        # (diffstd -> inf => uniform)
        self.diffstd = diffstd
        
        # criterion-prior-kappa: determines prior over criterion values,
        # the larger the kappa, the stronger cardinal criteria are expected
        # chose to implement as sqrt(kappa) to make scale comparable to 
        # std-parameters; chose not to implement as std to set uniform 
        # distribution over criteria at parameter=0 which is easier to favour
        # in the parameter prior
        self.cpsqrtkappa = cpsqrtkappa
        
        # expected spread of criterion observations
        self.critstd = critstd
        
        # expected spread of motion direction observations 
        # (previously called internal uncertainty as standard deviation)
        self.dirstd = dirstd
        
        # Standard deviation of noise added to criterion observation
        self.cnoisestd = cnoisestd
        
        # Standard deviation of noise added to motion direction observations
        self.dnoisestd = dnoisestd
            
        # Bound that needs to be reached before decision is made.
        # If collapsing, it's the initial value.
        self.bound = bound
            
        # Extent of collapse for bound, see boundfun.
        self.bstretch = bstretch
        
        # Shape parameter of the collapsing bound, see boundfun
        self.bshape = bshape
            
        # which ndt distribution to use?
        self.ndtdist = ndtdist
        
        # location of nondecision time distribution
        self.ndtloc = ndtloc
            
        # Spread of nondecision time.
        self.ndtspread = ndtspread
            
        # Probability of a lapse.
        self.lapseprob = lapseprob
            
        # Probability that a lapse will be timed out.
        self.lapsetoprob = lapsetoprob
        
        self._during_init = False
        
        
    def __str__(self):
        info = super(rotated_directions_diff, self).__str__()
        
        # empty line
        info += '\n'
        
        # model-specific parameters
        info += 'differences:\n'
        info += self.differences.__str__() + '\n'
        info += 'criteria:\n'
        info += self.criteria.__str__() + '\n'
        info += 'uses likes   : %4d' % self.use_liks + '\n'
        info += 'dt           : %8.3f' % self.dt + '\n'
        info += 'bound        : %8.3f' % self.bound + '\n'
        info += 'bstretch     : %7.2f' % self.bstretch + '\n'
        info += 'bshape       : %7.2f' % self.bshape + '\n'
        info += 'diffstd      : %6.1f' % self.diffstd + '\n'
        info += 'cpsqrtkappa  : %6.1f' % self.cpsqrtkappa + '\n'
        info += 'critstd      : %6.1f' % self.critstd + '\n'
        info += 'dirstd       : %6.1f' % self.dirstd + '\n'
        info += 'cnoisestd    : %6.1f' % self.cnoisestd + '\n'
        info += 'dnoisestd    : %6.1f' % self.dnoisestd + '\n'
        info += 'ndtdist      : %s'    % self.ndtdist + '\n'
        info += 'ndtloc       : %7.2f' % self.ndtloc + '\n'
        info += 'ndtspread    : %7.2f' % self.ndtspread + '\n'
        info += 'lapseprob    : %7.2f' % self.lapseprob + '\n'
        info += 'lapsetoprob  : %7.2f' % self.lapsetoprob + '\n'
        info += 'bias         : %7.2f' % self.bias + '\n'
        
        return info
    
    def gen_timecourses(self, trind):
        """Generate time courses of internal model variables.
        
        This re-implements the model mechanism in pure Python and is therefore
        not useful for large-scale model predictions. It's just a tool for 
        investigating the inference mechanisms implemented by the model for the
        parameter values currently stored in the model.
        
        Returns
        -------
        times  - the time points simulated by the model
        lpD    - log prior over differences (normalised) given choice
        lpCR   - log prior over criteria (unnormalised)
        lCR    - log likelihood over criteria for observing the criterion 
                 presented in the selected trial (unnormalised)
        lE_OM  - log evidence for motion directions after observing a motion 
                 direction sample (unnormalised)
        lE_DR  - log evidence for all combinations of difference and criterion
                 (unnormalised)
        lp_rot - log probability over clockwise rotation (normalised)
        """
        N = trind.size
        
        times = np.arange(0, self.S+1) * self.dt
        
        dirkappa = 1 / self.dirstd ** 2
        dnoisekappa = 1 / self.dnoisestd ** 2
        cnoisekappa = 1 / self.cnoisestd ** 2
        critkappa = 1 / self.critstd ** 2
        
        radcrit = to_rad(self.criteria)
        raddiff = to_rad(self.differences)
        
        cw = self.differences > 0
        acw = self.differences < 0
        
        # discretised half-normal prior over differences 
        # (given a particular choice)
        lpD = np.full((self.DF, 2), -np.inf)
        lpD[cw, 0] = - raddiff[cw] ** 2 / self.diffstd ** 2
        lpD[acw, 1] = - raddiff[acw] ** 2 / self.diffstd ** 2
        lpD -= logsumexp_2d(lpD, axis=0)
        
        lpCR = logsumexp_2d(
                np.c_[np.cos(2 * radcrit), 
                      np.cos(2 * (radcrit - np.pi / 2))] 
                * self.cpsqrtkappa ** 2, axis=1)[:, 0]
        
        lCR = np.zeros((self.CR, N))
        lE_OM = np.zeros((self.DF, self.CR, self.S + 1, N))
        lE_OM[:, :, 0, :] = np.nan
        lE_DR = np.zeros((self.DF, self.CR, self.S + 1, N))
        lp_rot = np.zeros((2, self.S + 1, N))
        lp_rot[:, 0, :] = np.log(np.r_[self.bias, 1 - self.bias])[:, None]
        for i, tr in enumerate(trind):
            # for some reason np.random.vonmises goes into very long 
            # computations for very large kappa (eg. cnoisestd=1e-12), so I
            # explicitly loop over trials and use random.vonmisesvariate
            lCR[:, i] = np.cos(2 * (
                    random.vonmisesvariate(
                            to_rad(self.Trials['criteria'][tr]),
                            cnoisekappa) % np.pi
                    - radcrit)) * critkappa
                    
            lE_DR[:, :, 0, i] = np.tile(lpCR + lCR[:, i], (self.DF, 1))
            
            for t in range(1, self.S+1):
                # sample observed motion direction
                o_dir = random.vonmisesvariate(
                        to_rad(self.Trials['directions'][tr]),
                        dnoisekappa)
                
                for cr in range(self.CR):
                    # compute log-evidences for observed direction
                    lE_OM_cr = np.zeros((self.DF, 2))
                    for df in range(self.DF):
                        lE_OM_cr[df, :] = np.cos(o_dir - (
                                radcrit[cr] + np.c_[0, np.pi] 
                                - raddiff[df])) * dirkappa - math.log(2)
                    
                    lE_OM_cr -= math.log(2)
                    
                    lE_OM[:, cr, t, i] = logsumexp_2d(lE_OM_cr, axis=1)[:, 0]
                    
                    lE_DR[:, cr, t, i] = (lE_DR[:, cr, t-1, i] 
                                          + self.dt * lE_OM[:, cr, t, i])
                    
                lE_D = logsumexp_2d(lE_DR[:, :, t, i], axis=1)[:, 0]
                
                lp_rot[0, t, i] = (logsumexp(lpD[cw, 0] + lE_D[cw])
                             + math.log(self.bias))
                lp_rot[1, t, i] = (logsumexp(lpD[acw, 1] + lE_D[acw])
                             + math.log(1 - self.bias))
                lp_rot[:, t, i] -= logsumexp(lp_rot[:, t, i])
                
            
        return times, lpD, lpCR, lCR, lE_OM, lE_DR, lp_rot
    
    
    def plot_example_timecourses(self, trind, dvtype='prob', with_bound=True,
                                 seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        times, lpD, lpCR, lCR, lE_OM, lE_DR, lp_rot = self.gen_timecourses(
                trind)
        
        labels = {self.toresponse[0]: 'ambiguous', 
                  self.choices[0]: 'clockwise', 
                  self.choices[1]: 'anti-clockwise'}
        cols = {self.toresponse[0]: 'k', 
                self.choices[0]: 'C0', 
                self.choices[1]: 'C1'}
        
        if dvtype == 'prob':
            dv = np.exp(lp_rot[0, :, :])
        elif dvtype == 'logprob':
            dv = lp_rot[0, :, :]
        elif dvtype == 'logprobdiff':
            dv = lp_rot[0, :, :] - lp_rot[1, :, :]
        
        fig, ax = plt.subplots()
        for c in np.r_[self.toresponse[0], self.choices]:
            ind = self.correct[trind] == c
            if ind.sum() > 0:
                lines = ax.plot(times, dv[:, ind], 
                                color=cols[c], alpha=.3)
                lines[0].set_label(labels[c])
        
        if with_bound:
            lower = 1 - self.bound
            upper = self.bound
            if dvtype == 'logprob':
                lower = np.log(lower)
                upper = np.log(upper)
            elif dvtype == 'logprobdiff':
                upper = np.log(upper / (1 - upper))
                lower = -upper
                
            line, = ax.plot(times, np.ones_like(times) * upper, 'k')
            line, = ax.plot(times, np.ones_like(times) * lower, 'k')
            line.set_label('bounds')
        
        ax.legend(loc='upper left')
        ax.set_xlabel('time (s)')
        if dvtype == 'prob':
            ax.set_ylabel('probability clockwise rotation')
        elif dvtype == 'logprob':
            ax.set_ylabel('log-probability clockwise rotation')
        elif dvtype == 'logprobdiff':
            ax.set_ylabel('log p(clockwise) - log p(anti-clockwise)')
        
        return fig, ax
    
    
    def plot_dt_ndt_distributions(self, mean=None, cov=None, pars=None, R=30):
        """Plot decision and non-decision time distributions separately.
        
        Uses either the parameters stored in the model, or the given parameter
        posterior. Decision time distribution will include lapses.
        """
        
        fig, ax = plt.subplots()
        
        if mean is not None:
            params = pars.sample_transformed(self.L * R, mean, cov)
            
            inds = [pars.names[pars.names == 'ndtloc'].index.values[0],
                    pars.names[pars.names == 'ndtspread'].index.values[0]]
        
        # sample from decision time distribution
        if mean is None:
            ndtloc = self.ndtloc
            ndtspread = self.ndtspread
            self.ndtloc = -30
            self.ndtspread = 0
            choices, rts = self.gen_response(np.arange(self.L), rep=R)
            self.ndtloc = ndtloc
            self.ndtspread = ndtspread
        else:
            ndtloc = params[:, inds[0]].copy()
            ndtspread = params[:, inds[1]].copy()
            params[:, inds[0]] = -30
            params[:, inds[1]] = 0
            choices, rts = self.gen_response_with_params(
                    np.tile(np.arange(self.L), R), params, pars.names)
            params[:, inds[0]] = ndtloc
            params[:, inds[1]] = ndtspread
            
        self.plot_response_distribution(choices, rts, ax=ax)
        
        # sample from non-decision time distribution
        if mean is None:
            if self.ndtdist == 'lognormal':
                rts = np.random.lognormal(self.ndtloc, self.ndtspread, 
                                          self.L * R)
            elif self.ndtdist == 'uniform':
                rts = (np.random.rand(self.L * R) * self.ndtspread
                       + np.exp(self.ndtloc))
        else:
            if self.ndtdist == 'lognormal':
                rts = np.array([np.random.lognormal(mu, sig) 
                                for (mu, sig) in params[:, inds]])
            elif self.ndtdist == 'uniform':
                rts = (np.random.rand(params.shape[0]) * params[:, inds[1]]
                       + np.exp(params[:, inds[0]]))
            
        choices = self.choices[np.zeros(self.L * R, dtype=int)]
        
        self.plot_response_distribution(choices, rts, alpha=0.5, ax=ax)
        
        ax.set_xlabel('time (s)')
        ax.set_ylabel('density')
        
        ax.legend([c[0] for c in ax.containers], 
                  ['dt for choice %d' % self.choices[0], 
                   'dt for choice %d' % self.choices[1], 
                   'ndt'])
        
        return fig, ax
    
    
    def estimate_memory_for_gen_response(self, N):
        """Estimate how much memory you would need to produce the desired 
           responses."""
        
        mbpernum = 8 / 1024 / 1024
        
        # (for input features + for input params + for output responses)
        return mbpernum * N * (self.S + self.P + 2)
    
    
    def gen_response(self, trind, rep=1):
        N = trind.size
        if rep > 1:
            trind = np.tile(trind, rep)
        
        choices, rts = self.gen_response_with_params(trind)
        
        if rep > 1:
            choices = choices.reshape((rep, N))
            rts = rts.reshape((rep, N))
        
        return choices, rts
        
        
    def gen_response_with_params(self, trind, params={}, parnames=None, 
                                 user_code=True):
        if parnames is None:
            assert( isinstance(params, dict) )
            pardict = params
        else:
            assert( isinstance(params, np.ndarray) )
            pardict = {name: params[:, ind] 
                       for ind, name in enumerate(parnames)}
                
        parnames = pardict.keys()
        
        # check whether any of the bound parameters are given
        if any(x in ['bound', 'bstretch', 'bshape'] for x in parnames):
            changing_bound = True
        else:
            changing_bound = False
        
        # get the number of different parameter sets, P, and check whether the 
        # given parameter counts are consistent (all have the same P)
        P = None
        for name in parnames:
            if not np.isscalar(pardict[name]):
                if P is None:
                    P = pardict[name].shape[0]
                else:
                    if P != pardict[name].shape[0]:
                        raise ValueError('The given parameter dictionary ' +
                            'contains inconsistent parameter counts')
        if P is None:
            P = 1
        
        # get the number of trials, N, and check whether it is consistent with 
        # the number of parameters P
        if np.isscalar(trind):
            trind = np.full(P, trind, dtype=int)
        N = trind.shape[0]
        if P > 1 and N > 1 and N != P:
            raise ValueError('The number of trials in trind and the ' +
                             'number of parameters in params does not ' + 
                             'fit together')
        
        NP = max(N, P)
        
        # if continuing would exceed the memory limit
        if self.estimate_memory_for_gen_response(NP) > self.memlim:
            # divide the job in smaller batches and run those
        
            # determine batch size for given memory limit
            NB = int(math.floor(NP / self.estimate_memory_for_gen_response(NP) 
                                * self.memlim))
            
            choices = np.zeros(NP, dtype=np.int8)
            rts = np.zeros(NP)
            
            remaining = NP
            firstind = 0
            while remaining > 0:
                index = np.arange(firstind, firstind + min(remaining, NB))
                if P > 1 and N > 1:
                    trind_batch = trind[index]
                    params_batch = extract_param_batch(pardict, index)
                elif N == 1:
                    trind_batch = trind
                    params_batch = extract_param_batch(pardict, index)
                elif P == 1:
                    trind_batch = trind[index]
                    params_batch = pardict
                else:
                    raise RuntimeError("N and P are not consistent.")
                    
                choices[index], rts[index] = self.gen_response_with_params(
                    trind_batch, params_batch, user_code=user_code)
                
                remaining -= NB
                firstind += NB
        else:
            # make a complete parameter dictionary with all parameters
            # this is quite a bit of waste of memory and should probably be recoded
            # more sensibly in the future, but for now it makes the jitted function
            # simple
            allpars = {}
            for name in self.parnames:
                if name in parnames:
                    allpars[name] = pardict[name]
                else:
                    allpars[name] = getattr(self, name)
                    
                if np.isscalar(allpars[name]) and N >= 1:
                    allpars[name] = np.full(N, allpars[name], dtype=float)
                elif allpars[name].shape[0] == 1 and N > 1:
                    allpars[name] = np.full(N, allpars[name], dtype=float)
            
            # select input for directions
            if self.use_liks:
                dirs = self.Trials['directions'][:, :, trind]
            else:
                dirs = self.Trials['directions'][trind]
                
            # call the compiled function
            choices, rts = self.gen_response_jitted(
                    dirs, self.Trials['criteria'][trind], allpars, 
                    changing_bound)
                
            # transform choices to those expected by user, if necessary
            if user_code:
                toresponse_intern = np.r_[-1, self.toresponse[1]]
                timed_out = choices == toresponse_intern[0]
                choices[timed_out] = self.toresponse[0]
                in_time = np.logical_not(timed_out)
                choices[in_time] = self.choices[choices[in_time]]
            
        return choices, rts
    
    
    def gen_response_jitted(self, trial_directions, trial_criteria, allpars, 
                            changing_bound):
        toresponse_intern = np.r_[-1, self.toresponse[1]]
            
        # for numba trial_directions needs to have a consistent size
        if trial_directions.ndim == 1:
            trial_directions = trial_directions[:, None, None, None]
        
        # call the compiled function
        choices, rts = gen_response_jitted_diff(
                trial_directions, trial_criteria, self.maxrt, toresponse_intern, 
                self.choices, self.dt, self.differences, self.criteria, 
                allpars['bias'], allpars['diffstd'], allpars['cpsqrtkappa'], 
                allpars['critstd'], allpars['dirstd'], allpars['cnoisestd'], 
                allpars['dnoisestd'], allpars['bound'], allpars['bstretch'], 
                allpars['bshape'], allpars['ndtloc'], allpars['ndtspread'], 
                allpars['lapseprob'], allpars['lapsetoprob'], changing_bound,
                0 if self.ndtdist == 'lognormal' else 1)
            
        return choices, rts
    
    
@jit(nopython=True, parallel=True)
def gen_response_jitted_diff(
        trdir, trcrit, maxrt, toresponse, choices, dt, differences, criteria,
        bias, diffstd, cpsqrtkappa, critstd, dirstd, cnoisestd, dnoisestd, bound, 
        bstretch, bshape, ndtloc, ndtspread, lapseprob, lapsetoprob, 
        changing_bound, ndtdist):
    
    CR = len(criteria)
    DF = len(differences)
    C = len(choices)
    
    cw = differences > 0
    acw = differences < 0
    
    radcrit = to_rad(criteria)
    raddiff = to_rad(differences)
    
    # unscaled densities for the von Mises mixture in criterion prior
    # I had to precompute these here, because numba breaks with an internal
    # error, if I put this code at the right place inside the parallel for loop
    rc0 = np.cos(2 * radcrit)
    rc90 = np.cos(2 * (radcrit - np.pi / 2))
    
    if trdir.shape[0] == DF:
        use_liks = True
        _, _, S, N = trdir.shape
    else:
        use_liks = False
        N = trdir.size
        S = int(math.ceil(maxrt / dt)) + 1
    
    choices_out = np.full(N, toresponse[0], dtype=np.int8)
    rts = np.full(N, toresponse[1], np.float64)
    
    # pre-compute collapsing bound
    boundvals = np.full(S, math.log(bound[0]))
    if bstretch[0] > 0:
        for t in range(S):
            boundvals[t] = math.log( boundfun((t+1.0) / maxrt, bound[0], 
                bstretch[0], bshape[0]) )
    
    for tr in prange(N):
        # is it a lapse trial?
        if random.random() < lapseprob[tr]:
            # is it a timed-out lapse trial?
            if random.random() < lapsetoprob[tr]:
                choices_out[tr] = toresponse[0]
                rts[tr] = toresponse[1]
            else:
                choices_out[tr] = random.randint(0, C-1)
                rts[tr] = random.random() * maxrt
        else:
            # discretised half-normal prior over differences 
            # (given a particular choice)
            lprior_D = np.full((DF, 2), -np.inf)
            lprior_D[cw, 0] = - raddiff[cw] ** 2 / diffstd[tr] ** 2
            lprior_D[acw, 1] = - raddiff[acw] ** 2 / diffstd[tr] ** 2
            Z = logsumexp_2d(lprior_D, axis=0)[0, :]
            for df in range(DF):
                lprior_D[df, :] -= Z
            
            # compute criterion prior as mixture of two von Mises distributions
            # centred on the cardinal orientations (0 and 90), 
            # ignoring normalising constants
            # implement as mixture of 3 densities centred on 0, 90 and 180, 
            # because von Mises wraps around at 360 degrees and not at 180
            # degrees what I need here
            cpkappa = cpsqrtkappa[tr] ** 2
            lpCR = logsumexp_2d(np.stack(
                    (rc0 * cpkappa, 
                     rc90 * cpkappa)).T, axis=1)[:, 0]
            
            # compute evidences for the criterion value by sampling a criterion
            # observation and determining its likelihood for each considered
            # criterion; note: modulo np.pi ensures that the sampled criteria
            # are in [0, 180) and the factor of 2 in np.cos accounts for the 
            # corresponding fact that criteria wrap around at 180 degree 
            # and not at 360
            o_crit = random.vonmisesvariate(
                    to_rad(trcrit[tr]), 1 / cnoisestd[tr]**2) % np.pi
            lCR = np.cos(2 * (o_crit - radcrit)) / critstd[tr]**2
            
            dirkappa_tr = 1 / dirstd[tr] ** 2
            dnoisekappa_tr = 1 / dnoisestd[tr] ** 2
            
            # initialise the log-evidence of observations with computed 
            # criteria evidences, dim will be DF x CR
            lE_DR = np.zeros((DF, CR))
            for df in range(DF):
                lE_DR[df, :] = lpCR + lCR
                
            # for all considered time points
            for t in range(S):
                # get current bound value
                if changing_bound:
                    # need to compute boundval from parameters in this trial
                    if bstretch[tr] == 0:
                        boundval = math.log(bound[tr])
                    else:
                        boundval = math.log( boundfun((t+1.0) / maxrt, 
                            bound[tr], bstretch[tr], bshape[tr]) )
                else:
                    # can use pre-computed bound value
                    boundval = boundvals[t]
                
                # initialise log-evidence for each difference, use -inf,
                # because you will sum exp(log-evidence) and you don't want
                # the initial value to contribute to the sum
                lE_D = np.full(DF, -np.inf)
                
                if use_liks:
                    # get log-evidence for current direction observation
                    # as provided as input
                    for cr in range(CR):
                        # although dirstd is only defined for the observation
                        # likelihood, I here also use it to scale the 
                        # pre-computed likelihoods to allow the model to reduce
                        # the influence of observations on the decision 
                        # variable in a compatible way (large dirstd reduces
                        # influence of observations)
                        lE_DR[:, cr] += dt * (logsumexp_2d(
                                np.stack((trdir[:, cr, t, tr] * dirkappa_tr, 
                                          trdir[:, cr + CR, t, tr]
                                          * dirkappa_tr)).T, axis=1))[:, 0]
                        lE_D = logsumexp_2d(np.stack((lE_D, lE_DR[:, cr])).T,
                                            axis=1)[:, 0]
                else:
                    # compute log-evidence for current direction observation by
                    # sampling a corresponding observation
                    o_dir = random.vonmisesvariate(
                            to_rad(trdir[tr, 0, 0, 0]), dnoisekappa_tr)
                    
                    for cr in range(CR):
                        # compute log-evidences for observed direction
                        lE_OM = np.zeros((DF, 2))
                        for df in range(DF):
                            lE_OM[df, 0] = np.cos(o_dir - (
                                    radcrit[cr] - raddiff[df])) * dirkappa_tr
                            lE_OM[df, 1] = np.cos(o_dir - (
                                    radcrit[cr] + np.pi - raddiff[df])) * dirkappa_tr
                        
                        lE_OM -= math.log(2)
                        
                        lE_DR[:, cr] += dt * logsumexp_2d(lE_OM, axis=1)[:, 0]
                        
                        lE_D = logsumexp_2d(np.stack((lE_D, lE_DR[:, cr])).T,
                                            axis=1)[:, 0]
                        
                # log-probability of clockwise and anti-clockwise rotation
                # (needs to be renormalised)
                lp_rot = np.zeros(2)
                lp_rot[0] = (logsumexp(lprior_D[cw, 0] + lE_D[cw])
                             + math.log(bias[tr]))
                lp_rot[1] = (logsumexp(lprior_D[acw, 1] + lE_D[acw])
                             + math.log(1 - bias[tr]))
                lp_rot -= logsumexp(lp_rot)
                
                if lp_rot[0] >= boundval or lp_rot[1] >= boundval:
                    if ndtdist == 0:
                        ndt = random.lognormvariate(ndtloc[tr], ndtspread[tr])
                    else:
                        ndt = (random.random() * ndtspread[tr]
                               + math.exp(ndtloc[tr]))
                        
                    # add 1 to t because t starts from 0
                    rts[tr] = (t+1) * dt + ndt
                    
                    if lp_rot[0] >= boundval:
                        choices_out[tr] = 0
                    else:
                        choices_out[tr] = 1
                        
                    break
                
            if rts[tr] > maxrt:
                choices_out[tr] = toresponse[0]
                rts[tr] = toresponse[1]
    
    return choices_out, rts


@vectorize([float64(int16), float64(int32), float64(int64), float64(float64)],
           nopython=True, cache=True)
def to_rad(degree):
    return float(degree) / 180 * np.pi


@vectorize([int16(int16, int16), int32(int32, int32), int64(int64, int64)], 
           nopython=True, cache=True)
def compute_minimal_differences(direct, crit):
    """Computes minimal difference between criteria and directions so that 
       negative differences correspond to anit-clockwise rotation and positive 
       ones correspond to clockwise rotation.
       
       The difficulty here is that the criteria simultaneously represent upward
       and downward directions and we are only interested in the smallest 
       differences among the two interpretations, but the differences also need
       to have the correct sign.
    """
    
    # first interpretation of criterion as upward direction
    diff1 = crit - direct
    if diff1 <= -180:
        diff1 += 360
    
    # second interpretation of criterion as downward direction
    diff2 = (crit + 180) - direct
    if diff2 >= 180:
        diff2 -= 360
    
    return diff1 if np.abs(diff1) <= np.abs(diff2) else diff2


@jit(nopython=True, cache=True)
def get_rotations(directions, crit):
    cw = np.zeros_like(directions, dtype=np.bool8)
    acw = np.zeros_like(directions, dtype=np.bool8)
    ontop = np.zeros_like(directions, dtype=np.bool8)
    between = np.zeros_like(directions, dtype=np.bool8)
    
    # convert to degrees and integers so that comparison operations become
    # exact
    directions = np.rint(directions / math.pi * 180).astype(np.int16)
    crit = round(crit / math.pi * 180)
    
    # rotate both criterion and directions to horizontal criterion (setting 
    # criterion to 0)
    directions = directions - crit
    
    # some directions may now be negative, the modulo gets their positive 
    # counterpart in [0, 360]
    directions = directions % 360
    
    for d, direction in enumerate(directions):
        # the direction is clockwise rotated with respect to the criterion,
        # if the angle stops in quadrant II or IV
        if direction == 0 or direction == 180:
            ontop[d] = True
        elif direction == 90 or direction == 270:
            between[d] = True
        elif ((direction > 90 and direction < 180) or 
              (direction > 270 and direction < 360)):
            cw[d] = True
        else:
            acw[d] = True
        
    return cw, acw, ontop, between


@jit(nopython=True, cache=True)
def logsumexp(logvals):
    mlogv = logvals.max()
    
    # bsxfun( @plus, mlogp, log( sum( exp( bsxfun(@minus, logp, mlogp) ) ) ) );
    logsum = mlogv + np.log( np.sum( np.exp(logvals - mlogv) ) )
    
    return logsum


@jit(nopython=True, cache=True)
def logsumexp_2d(logvals, axis=0):
    shape = logvals.shape
    
    assert len(shape) == 2, 'logvals in logsumexp_3d has to be 2d!'
    
    if axis == 0:
        logsum = np.zeros((1, shape[1]))
        for i1 in range(shape[1]):
            logsum[0, i1] = logsumexp(logvals[:, i1])
    elif axis == 1:
        logsum = np.zeros((shape[0], 1))
        for i0 in range(shape[0]):
            logsum[i0, 0] = logsumexp(logvals[i0, :])
    else:
        raise ValueError("Argument 'axis' has illegal value in "
                         "logsumexp_3d!")
    
    return logsum


@jit(nopython=True, cache=True)
def boundfun(tfrac, bound, bstretch, bshape):
    tshape = tfrac ** -bshape
    
    return 0.5 + (1 - bstretch) * (bound - 0.5) - ( bstretch * (bound - 0.5) *
        (1 - tshape) / (1 + tshape) );
                  
    
def extract_param_batch(pardict, index):
    newdict = {}
    for parname, values in pardict.items():
        if values.ndim == 2:
            newdict[parname] = values[index, :]
        else:
            newdict[parname] = values[index]
            
    return newdict