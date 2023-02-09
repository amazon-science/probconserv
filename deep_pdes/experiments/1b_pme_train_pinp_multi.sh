#!/bin/bash
EXPERIMENT=1b_pme_var_m
# for l in 1en2 1en1 1e1 1e2 1e6
for l in 1e6
do
	echo "Training ANP+SoftC with lambda="${l}
	python train.py +experiments=$EXPERIMENT +train=${EXPERIMENT}_pinp_${l}
done
