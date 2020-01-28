#!/usr/bin/env bash

conda install -c anaconda swig
conda install pip
conda install tensorflow-gpu==1.15.0
pip install -e .
``