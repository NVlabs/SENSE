#!/bin/bash

conda env create -f sense_conda_environment.yml

cd sense/lib/correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
python setup.py install --user

