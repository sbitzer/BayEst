#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:35:45 2018

@author: bitzer
"""
import os
import numpy as np
import pandas as pd
import helpers
import re

import hddm


#%% options
exclude_to = True
censor_late = True
S = 6000
B = 2000
PS = 1000
p_outlier = 0.05
depends_on = {'v': 'easy'}
bias = True

resfile = os.path.join(helpers.resultsdir, 'hddm',
                       pd.datetime.now().strftime('hddm_result_%Y%m%d%H%M.h5'))

with pd.HDFStore(resfile, mode='w', complevel=7, complib='blosc') as store:
    store['scalar_opt'] = pd.Series(dict(
            exclude_to=exclude_to, censor_late=censor_late, B=B, S=S, PS=PS,
            p_outlier=p_outlier, bias=bias))
    store['depends_on'] = pd.Series(depends_on)


#%% some helpers for data storage
is_easy = lambda s: re.match(r'wfpt\((\w+)\.\d\)', s).group(1) == 'True'
get_stim = lambda s: int(s[-2:-1])


#%% loop over subjects
for sub in helpers.find_available_subjects():
    print("\ninferring for subject %2d ..." % sub)
    
    data = helpers.load_subject(sub, exclude_to=exclude_to, 
                                censor_late=censor_late)
    
    stim = data.response.copy().astype(int)
    err = data.error.astype(bool)
    stim[err] = -stim[err]
    
    hddm_data = pd.DataFrame(
            [data.RT.values, 
             (data.response + 1) / 2,
             (stim + 1) / 2, 
             sub * np.ones(data.shape[0]),
             data.easy.values], 
            index=['rt', 'response', 'stim', 'subj_idx', 'easy']).T
    
    hddm_data = hddm_data.astype(dict(rt=float, response=int, subj_idx=int, 
                                      stim=int, easy=bool))
    
    #%% create and sample from model
    model = hddm.HDDMStimCoding(hddm_data, bias=bias, stim_col='stim', 
                                split_param='v', depends_on=depends_on,
                                p_outlier=p_outlier, is_group_model=False)
    
    # produced error when called with bias=True
    #model.find_starting_values()
    
    model.sample(S, burn=B)
    
    samples = model.get_traces()
    samples.index = pd.MultiIndex.from_arrays(
            [np.full(S-B, sub), samples.index], names=['subject', 'sample'])
    
    with pd.HDFStore(resfile, mode='a', complib='blosc', complevel=7) as store:
        store.append('samples', samples)
    
    
    #%% posterior predictive simulations
    ppc_data = hddm.utils.post_pred_gen(model, samples=PS)
    
    ppc_data['rt'] = ppc_data.rt.abs()
    ppc_data['easy'] = ppc_data.index.get_level_values('node').map(
            is_easy).astype(bool)
    # for some reason this didn't work when I tried it with the series above
    ppc_data['easy'] = ppc_data.easy.astype(bool)
    ppc_data['stim'] = ppc_data.index.get_level_values('node').map(get_stim)
    ppc_data.index = ppc_data.index.droplevel('node')
    ppc_data['subject'] = sub
    ppc_data.set_index('subject', append=True, inplace=True)
    ppc_data = ppc_data.reorder_levels([2, 0, 1])
    
    with pd.HDFStore(resfile, mode='a', complib='blosc', complevel=7) as store:
        store.append('ppc_data', ppc_data)