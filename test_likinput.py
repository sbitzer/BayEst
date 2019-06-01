#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:25:40 2019

@author: bitzer
"""
import math

import numpy as np
import pytest

import rotated_directions as rd

@pytest.fixture(scope='module')
def directions():
    return (np.linspace(0, 1 - 1/8, 8)
            * 2 * math.pi - math.pi)

@pytest.fixture(scope='module')
def model(directions):
    Trials = np.zeros((100, directions.size, 1))
    return rd.rotated_directions(Trials, dt=0.02, maxrt=1.9, directions=directions,
                    intstd=0.5, noisestd=1e-12, lapseprob=0.0)

@pytest.fixture(params=['nonzero', 'onelarge'])
def Trials(model, request):
    if request.param == 'nonzero':
        Tr = np.zeros((model.S, model.D, model.D))
        for d in range(model.D):
            Tr[:, d, d] = 1
    elif request.param == 'onelarge':
        Tr = np.ones((model.S, model.D, model.D))
        for d in range(model.D):
            Tr[:, d, d] = 2

    return Tr

def test_lik(model, Trials):
    model.Trials = Trials
    model.criteria = model.criteria[0]

    ch, rt = model.gen_response(np.arange(model.L))

    cw, acw, ontop, between = rd.get_rotations(
            model.directions, model.criteria[0])

    for d in range(model.D):
        if cw[d]:
            assert ch[d] == model.choices[0]
        elif acw[d]:
            assert ch[d] == model.choices[1]
        else:
            assert ch[d] == model.toresponse[0]

@pytest.fixture(params=['equal', 'cancel'])
def Trials_timeout(model, request):
    if request.param == 'equal':
        Tr = np.ones((model.S, model.D, 1))
    elif request.param == 'cancel':
        cw, acw, ontop, between = rd.get_rotations(
                model.directions, model.criteria[0])
        cw = np.flatnonzero(cw)
        acw = np.flatnonzero(acw)

        L = min(cw.size, acw.size)

        Tr = np.ones((model.S, model.D, L))
        for tr in range(L):
            Tr[:, cw[tr], tr] = 2
            Tr[:, acw[tr], tr] = 2

    return Tr

def test_lik_timeouts(model, Trials_timeout):
    model.Trials = Trials_timeout
    model.criteria = model.criteria[0]

    ch, rt = model.gen_response(np.arange(model.L))

    assert np.all(ch == model.toresponse[0])

def test_lik_speed(model):
    Tr = np.zeros((model.S, model.D, 5))
    for tr in range(Tr.shape[2]):
        Tr[:, 1, tr] = tr + 1

    model.Trials = Tr
    model.criteria = model.criteria[0]

    ch, rt = model.gen_response(np.arange(model.L))

    assert np.all(np.diff(rt) < 0)
