#!/bin/bash

export DATASET=webarena

model="gpt-4o-mini"
value_function="gpt-4o-mini"

max_depth=2  # max_depth=4 means 5 step lookahead
max_steps=5
branching_factor=3
vf_budget=20
agent="world_model" # change this to "baseline" to run the baseline without world_model
world_model_training=True
value_model_training=True
world_model_type="finetuned" # finetuned or prompt
value_model_type="finetuned" # finetuned or prompt
next_state_format="description_with_tao"
result_dir="outputs/wma_${next_state_format}_format"
instruction_path="agent/prompts/jsons/p_cot_id_actree_2s_no_na.json"
state_prediction_prompt_path="agent/prompts/jsons/state_prediction/text_only_${next_state_format}_format.json"
value_function_prompt_path="agent/prompts/jsons/value_function/text_only_value_function.json"
world_model_name=""
world_model_url=""
value_model_name=""
value_model_url=""


### Code to run the experiments
function run_job() {
    local start_idx=$1
    local end_idx=$2
    local job_num=$3
    if [ -f logs/wma_${next_state_format}_format_job_${job_num}.log ]; then
        echo "----------------------------------------" >> logs/wma_${next_state_format}_format_job_${job_num}.log
        echo "New log entry started at $(date)" >> logs/wma_${next_state_format}_format_job_${job_num}.log
        echo "----------------------------------------" >> logs/wma_${next_state_format}_format_job_${job_num}.log
    else
        touch logs/wma_${next_state_format}_format_job_${job_num}.log
    fi
    nohup python run_w_world_model.py \
    --instruction_path $instruction_path \
    --test_start_idx $start_idx \
    --test_end_idx $end_idx \
    --model $model \
    --agent_type $agent   --max_depth $max_depth  --branching_factor $branching_factor  --vf_budget $vf_budget   \
    --result_dir $result_dir \
    --test_config_base_dir=config_files/wa/test_webarena \
    --repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
    --top_p 0.95   --temperature 1.0  --max_steps $max_steps --value_function $value_function\
    --state_prediction_prompt_path $state_prediction_prompt_path --value_function_prompt_path $value_function_prompt_path --total_indices $total_indices\
    $( [ "$world_model_training" = True ] && echo "--world_model_training" ) \
    $( [ "$world_model_training" = True ] && echo "--world_model_name $world_model_name" ) \
    $( [ "$world_model_training" = True ] && echo "--world_model_url $world_model_url" ) \
    $( [ "$value_model_training" = True ] && echo "--value_model_training" ) \
    $( [ "$value_model_training" = True ] && echo "--value_model_name $value_model_name" ) \
    $( [ "$value_model_training" = True ] && echo "--value_model_url $value_model_url" ) \
    >> logs/wma_${next_state_format}_format_job_${job_num}.log 2>&1 &
}
total_indices=10
indices_per_thread=1


current_start=0
job_count=0
while [ "$current_start" -lt "$total_indices" ]; do
    current_end=$((current_start + indices_per_thread))
    if [ "$current_end" -gt "$total_indices" ]; then
        current_end=$total_indices
    fi
    ((job_count++))

    # Run the job
    run_job $current_start $current_end $job_count

    # Increment start index for next job
    current_start=$current_end
done

### Wait for all jobs to complete
wait
