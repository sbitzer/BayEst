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
from numba import jit, prange
from warnings import warn
from rtmodels import rtmodel
import matplotlib.pyplot as plt


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
        
        if np.isscalar(crit):
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
                
    def gen_correct(self):
        """Tries to figure out correct choice from directions and criteria."""
        
        correct = np.zeros(self.L)
        for tr in range(self.L):
            if self.use_features:
                # determine average direction in trial from stored features
                direction = np.atan2(np.sin(self._Trials[:, tr]).sum(), 
                                     np.cos(self._Trials[:, tr]).sum())
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
        if self._Trials.ndim == 1:
            return False
        else:
            return True
    
    @property
    def Trials(self):
        """Trial information used by the model.
        
        Either a 1D (use_features=False), or 2D (use_features=True) numpy array.
        When 1D, Trials contains the code of the correct choice in that trial.
        When 2D, Trials contains the stream of feature values that the subject
        may have seen in all the trials of the experiment. Then,
            S, L = Tials.shape
        where S is the length of the sequence in each trial
        """
        if self.use_features:
            return self._Trials
        else:
            return self.directions[self._Trials]
        
    @Trials.setter
    def Trials(self, Trials):
        # ensure that Trials is array and is within [0, 2*pi]
        Trials = np.array(Trials) % (2 * np.pi)
        
        if Trials.ndim == 2:
            self._Trials = Trials
            S, self._L = Trials.shape
        elif Trials.ndim == 1:
            # check that Trials only contains valid directions
            if np.all(np.isin(np.unique(Trials), self.directions)):
                # transform to indices into self.directions
                self._Trials = np.array([
                        np.flatnonzero(self.directions == i)[0] 
                        for i in Trials])
                self._L = len(Trials)
            else:
                raise ValueError('Trials may only contain valid choice codes' +
                                 ' when features are not used.')
        else:
            raise ValueError('Trials has unknown format, please check!')
        
        if not self._during_init:
            self.gen_correct()
        
    
    @property
    def S(self):
        """Number of time steps maximally simulated by the model."""
        if self.Trials.ndim == 2:
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
                 lapsetoprob=0.1, ndtdist='lognormal', **rtmodel_args):
        super(rotated_directions, self).__init__(**rtmodel_args)
            
        self._during_init = True
        
        self.name = 'Discrete directions model'
        
        # Time resolution of model simulations.
        self.dt = dt
        
        # set directions that are estimated
        self.directions = directions
        
        # Trial information used by the model.
        self.Trials = np.array(Trials)
        
        # criterion orientations
        self.criteria = criteria
        
        # figure out the correct choice in each trial
        self.gen_correct()
        
        # check whether design is balanced (and warn if not)
        self.check_balance()
        
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
    
    
    def plot_example_timecourses(self, trind, dvtype='prob', with_bound=True):
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
        
        fig, ax = plt.subplots()
        for c in np.r_[self.toresponse[0], self.choices]:
            ind = self.correct[trind] == c
            if ind.sum() > 0:
                lines = ax.plot(times, dv[:, ind], 
                                color=cols[c], alpha=.3)
                lines[0].set_label(labels[c])
        
        if with_bound:
            line, = ax.plot(times, np.ones_like(times) * self.bound, 'k')
            line, = ax.plot(times, np.ones_like(times) - self.bound, 'k')
            line.set_label('bounds')
        
        ax.legend(loc='upper left')
        ax.set_xlabel('time (s)')
        if dvtype == 'prob':
            ax.set_ylabel('probability clockwise rotation')
        elif dvtype == 'logprob':
            ax.set_ylabel('log-probability clockwise rotation')
        
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
            NB = math.floor(NP / self.estimate_memory_for_gen_response(NP) *
                            self.memlim)
            
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
            if self.use_features:
                features = self.Trials[:, trind]
            else:
                features = self.directions[self._Trials[trind]]
                features = np.tile(features, (self.S, 1))
                
            # call the compiled function
            choices, rts = self.gen_response_jitted(
                    features, allpars, self.criteria[trind], changing_bound)
                
            # transform choices to those expected by user, if necessary
            if user_code:
                toresponse_intern = np.r_[-1, self.toresponse[1]]
                timed_out = choices == toresponse_intern[0]
                choices[timed_out] = self.toresponse[0]
                in_time = np.logical_not(timed_out)
                choices[in_time] = self.choices[choices[in_time]]
            
        return choices, rts
        
        
    def gen_response_jitted(self, features, allpars, criteria, changing_bound):
        toresponse_intern = np.r_[-1, self.toresponse[1]]
            
        # call the compiled function
        choices, rts = gen_response_jitted_dir(
                features, self.maxrt, toresponse_intern, 
                self.choices, self.dt, self.directions, criteria, 
                allpars['prior'], allpars['bias'], allpars['noisestd'], 
                allpars['intstd'], allpars['bound'], allpars['bstretch'], 
                allpars['bshape'], allpars['ndtloc'], allpars['ndtspread'], 
                allpars['lapseprob'], allpars['lapsetoprob'], changing_bound,
                0 if self.ndtdist == 'lognormal' else 1)
            
        return choices, rts

@jit(nopython=True, parallel=True)
def gen_response_jitted_dir(
        features, maxrt, toresponse, choices, dt, directions, criteria,
        prior, bias, noisestd, intstd, bound, bstretch, bshape, ndtloc, 
        ndtspread, lapseprob, lapsetoprob, changing_bound, ndtdist):
    
    D = len(directions)
    C = len(choices)
    S, N = features.shape
    
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
                
                # add noise to feature
                noisy_feature = random.vonmisesvariate(features[t, tr],
                        1 / noisestd[tr]**2)
                
                # compute log-probabilities of directions
                for d in range(D):
                    logpost[d] += dt * (math.cos(noisy_feature - directions[d]) 
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