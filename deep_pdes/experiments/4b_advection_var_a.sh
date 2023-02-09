#!/bin/bash
#set -o errexit

EXPERIMENT=4b_advection_var_a

python generate.py +experiments=$EXPERIMENT
python train.py +experiments=$EXPERIMENT +train=${EXPERIMENT}_anp
python train.py +experiments=$EXPERIMENT +train=${EXPERIMENT}_pinp
python analyze.py +experiments=$EXPERIMENT
