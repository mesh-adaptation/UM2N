#!/bin/bash

# python run_train_baselines.py -config MRT_miniset_new
# python run_train_baselines.py -config PIMRT_miniset_new
# python run_train_baselines.py -config M2N_miniset_new
# python run_train_baselines.py -config M2N_enhance_miniset_new

# python run_train_baselines.py -config MRT_largeset
# python run_train_baselines.py -config PIMRT_largeset
# python run_train_baselines.py -config M2N_largeset
# python run_train_baselines.py -config M2N_enhance_largeset


python run_train_baselines.py -config M2N_enhance_largeset_monitor_only
python run_train_baselines.py -config MRT_largeset_monitor_only