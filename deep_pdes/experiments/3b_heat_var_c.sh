#!/bin/bash
#set -o errexit

EXPERIMENT=3b_heat_var_c

python generate.py +experiments=$EXPERIMENT
python train.py +experiments=$EXPERIMENT +train=${EXPERIMENT}_anp
python train.py +experiments=$EXPERIMENT +train=${EXPERIMENT}_pinp
python analyze.py +experiments=$EXPERIMENT
