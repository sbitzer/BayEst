#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# for adding conda to the path
source ~/.bashrc

SNL_NAME=snl_t2

# where is the behavioural data on this machine?
DATA_DIR_PREFIX="/dune/Experiments/DR-BayAtt-2"
#DATA_DIR_PREFIX="/home/bitzer/ZIH/projects/DraganRangelov/data/behaviour"

# setting up data and results directories as needed in python code
mkdir -p data
ln -s $DATA_DIR_PREFIX data/behaviour

mkdir -p inf_results/behaviour/snl/rotated_directions

conda create -c conda-forge -n $SNL_NAME python=2.7 mkl=2017 mkl-service \
    theano ipython scipy pandas pytables numba seaborn gxx_linux-64 pytest

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

# add path to conda's g++ to theano
echo '[global]' >> ~/.theanorc
echo $CXX >> ~/.theanorc
