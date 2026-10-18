#!/bin/bash

dataset=""
trajectory_path="trajectory"
output_path="output"
run_index=5
start=0
end=712 # num of collected trajectories

python dataset_construction/src/train_data_constructor.py \
        --dataset $dataset \
        --trajectory_path $trajectory_path \
        --output_path $output_path \
        --run_index $run_index \
        --start $start \
        --end $end