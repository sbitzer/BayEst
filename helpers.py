#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:06:41 2018

@author: bitzer
"""

import os
import pandas as pd
import numpy as np

datadir = "../data/behaviour/Data"

choices = dict(left=-1, right=1)
toresponse = [0, 3]

def load_subject(sub):
    file = os.path.join(datadir, 's%02d_main_data.txt' % sub)
    
    data = pd.read_csv(file, '\t')
    
    assert np.all(data.dotCoh == data.dotCoh[0]), ("subject has differing dot "
                                                   "coherences!")
    
    data.response = data.response.map(choices, 'ignore')
    
    timeouts = data.RT.isna()
    data.loc[timeouts, 'RT'] = toresponse[1]
    data.loc[timeouts, 'response'] = toresponse[0]
    
    data['easy'] = np.abs(data.tarDir - data.critDir) > 20
    
    # ensure that directions are in valid ranges
    data.tarDir = data.tarDir % 360
    data.critDir = data.critDir % 180
    
    # copy so that not the full dataframe is kept in memory, because the 
    # returned object is otherwise only a reference to the full dataframe
    return data[['easy', 'tarDir', 'critDir', 'RT', 'response', 'error']
                ].copy()