#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# for adding conda to the path
source ~/.bashrc

SNL_NAME=snl_t2

conda create -c conda-forge -n $SNL_NAME python=2.7 mkl=2017 theano ipython scipy pandas pytables numba seaborn
source activate $SNL_NAME

git clone git@github.com:sbitzer/BayEst.git

git clone git@github.com:sbitzer/rtmodels.git
cd rtmodels
python setup.py develop
cd ..

git clone git@github.com:sbitzer/pyEPABC.git
cd pyEPABC
python setup.py develop
cd ..

# snl_package must already exist (copy from somewhere)
cd snl_package
python setup.py develop
cd ..