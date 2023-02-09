#!/bin/bash
#set -o errexit

EXPERIMENT=5b_burgers_var_a

python generate.py +experiments=$EXPERIMENT
python train.py +experiments=$EXPERIMENT +train=${EXPERIMENT}_pinp
python train.py +experiments=$EXPERIMENT +train=${EXPERIMENT}_anp
python analyze.py +experiments=$EXPERIMENT
