#!/bin/bash

dataset_path="dataset/sample_webarena_acctree.json"
model="gpt-4o-mini"
output_path="dataset"
prompt_path="dataset_construction/prompts/acctree.json"
api_key=""

python dataset_construction/src/annotation_for_acctree_rationale.py \
    --dataset_path $dataset_path \
    --output_path $output_path \
    --prompt_path $prompt_path \
    --api_key $api_key \
    --model $model \
    --batch_size 300 \
    --debug
