#!/usr/bin/env bash

nohup python3 train.py --dataset encoded_datasets/systems3.npz --run_name systems_3x3_774M --model_name 774M --sample_every 100 --sample_num 1 --save_every 100 --restore_from fresh &
