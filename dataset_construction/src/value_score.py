import argparse
import json

from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset/sample_mind2web_acctree.json")
    parser.add_argument("--output_path", type=str, default="dataset/sample_mind2web_acctree_with_value_score.json")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def calculate_value_score(args):
    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)

    if args.debug:
        dataset = dataset[:10]
    else:
        dataset = dataset

    # Group items by confirmed_task and find the max target_action_index for each group
    max_target_action_index_per_task = {}
    for item in dataset:
        task = item['confirmed_task']
        if task not in max_target_action_index_per_task:
            max_target_action_index_per_task[task] = int(item['target_action_index'])
        else:
            max_target_action_index_per_task[task] = max(max_target_action_index_per_task[task], int(item['target_action_index']))
    
    # Create a copy of the dataset and add 'value_score'
    updated_dataset = []
    for item in dataset:
        task = item['confirmed_task']
        max_target_action_index = max_target_action_index_per_task[task] + 1
        new_item = item.copy()
        new_item['value_score'] = str(round(float((int(item['target_action_index']) + 1) / (max_target_action_index + 1)), 2))
        updated_dataset.append(new_item)

    return updated_dataset


if __name__ == "__main__":
    # Example usage
    args = parse_args()
    updated_dataset = calculate_value_score(args)

    with open(args.output_path, "w") as f:
        json.dump(updated_dataset, f, indent=4, ensure_ascii=False)
    



