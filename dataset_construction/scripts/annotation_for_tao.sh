#!/bin/bash

dataset_path="dataset/sample_webarena_acctree.json"
output_path="dataset"

python dataset_construction/src/annotation_for_tao_torch.py \
        --dataset_path $dataset_path \
        --output_path $output_path \
        --use_torch \
        --debug