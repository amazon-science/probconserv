EXPERIMENT=2b_stefan_var_p

python generate.py +experiments=$EXPERIMENT
python train.py +experiments=$EXPERIMENT +train=${EXPERIMENT}_anp
python train.py +experiments=$EXPERIMENT +train=${EXPERIMENT}_pinp
python analyze.py +experiments=$EXPERIMENT
