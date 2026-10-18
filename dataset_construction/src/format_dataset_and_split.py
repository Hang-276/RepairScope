import os
import argparse
import json

from datasets import load_dataset
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--format", type=str, choices=["description_with_tao", "acctree"])
parser.add_argument("--prompt_path", type=str)
parser.add_argument("--output_path", type=str)
args = parser.parse_args()

with open(args.dataset_path, "r") as f:
    prev_dataset = json.load(f)

with open(os.path.join(args.prompt_path, f"text_only_{args.format}_format.json"), "r") as f:
    instruction = json.load(f)['intro']

formatted_dataset = []
for i in range(len(prev_dataset)):
    instance = prev_dataset[i]
    input_text = f"URL: {instance['url']}\n OBJECTIVE: {instance['objective']}\n PREVIOUS ACTION: {instance['previous_actions']}\n CURRENT OBSERVATION: {instance['observation']}\n CURRENT ACTION: {instance['current_action']}"
    
    if args.format == "acctree":
        output = "[Rationale] "+instance['next_state_rationale']+"\n[Next State] "+instance[f'next_state_{args.format}']
    else:
        output = "[Rationale] "+instance['rationale']+"\n[Next State] "+instance[f'next_state_{args.format}']

    formatted_instance = {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }
    
    formatted_dataset.append(formatted_instance)

# Calculate the split point for 8:2 ratio
split_point = int(len(formatted_dataset) * 0.8)

# Split the dataset
train_set = formatted_dataset[:split_point]
val_set = formatted_dataset[split_point:]

with open(os.path.join(args.output_path, f"sample_formatted_{args.format}_dataset.json"), "w") as f:
    json.dump({"train": train_set, "validation": val_set}, f, indent=4, ensure_ascii=False)