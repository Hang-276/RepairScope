#!/bin/bash

dataset="dataset/sample_description_with_tao_annotated.json"
format="description_with_tao" # ["description_with_tao", "acctree"]

output_path="dataset"
prompt_path="agent/prompts/jsons/state_prediction"

python dataset_construction/src/format_dataset_and_split.py \
        --dataset $dataset \
        --output_path $output_path \
        --prompt_path $prompt_path \
        --format $format
