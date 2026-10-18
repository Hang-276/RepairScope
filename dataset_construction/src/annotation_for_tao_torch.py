import re
import json
import torch
import time
import os
import argparse
import pygmtools as pygm

from multiprocessing import Pool, cpu_count
from multiprocessing import Value, Lock
from munkres import Munkres
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

pygm.set_backend("pytorch")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--use_torch", action="store_true", help="Use torch version")
    return parser.parse_args()


def parse_line(line):
    # Extract the backend ID, role, and content
    match = re.match(r'\[(\d+)\]\s*(\S+)\s*(.*)', line)
    if match:
        backend_id = int(match.group(1))
        role = match.group(2)
        content = match.group(3)
        return backend_id, role, content
    else:
        return None, None, None


def convert_observation_to_dict(obs):
    obs_dict = {}
    for line in obs.split("\n"):
        line = line.replace("\t", "")
        backend_id, role, content = parse_line(line)
        if not backend_id:
            continue
        obs_dict[backend_id] = {
            "backend_id": backend_id,
            "text": line,
            "role": role,
            "content": content
        }
    return obs_dict
    

def get_TaO(obs1, obs2, action_str, use_torch_version):
    obs1_dict = convert_observation_to_dict(obs1)
    obs2_dict = convert_observation_to_dict(obs2)
    if action_str != "None":
            
        # Find new, deleted, and updated elements
        if use_torch_version:
            new_items, deleted_items, updated_items = compare_observations_torch(obs1_dict, obs2_dict)
        else:
            new_items, deleted_items, updated_items = compare_observations(obs1_dict, obs2_dict)
        
        if len(new_items) == 0 and len(deleted_items) == 0 and len(updated_items) == 0:
            return "Previous Action did not make any changes.", None
        else:
            tao_str = ""
            # process new items 
            tao_str += "New items:\n"
            new_items_str = ""
            if len(new_items) == 0:
                tao_str += "None\n"
                new_items_str += "None"
            else:
                for v in new_items:
                    tao_str += v['text'] + "\n"
                    new_items_str += v['text'] + "\n"
            
            # process deleted items
            tao_str += "\nDeleted items:\n"
            deleted_items_str = ""
            if len(deleted_items) == 0:
                tao_str += "None\n"
                deleted_items_str += "None"
            else:
                for v in deleted_items:
                    tao_str += v['text'] + "\n"
                    deleted_items_str += v['text'] + "\n"
            
            # process updated items
            tao_str += "\nUpdated items:\n"
            updated_items_str = ""
            if len(updated_items) == 0:
                tao_str += "None\n"
                updated_items_str += "None"
            else:
                for v in updated_items:
                    tao_str += v['text'] + "\n"
                    updated_items_str += v['text'] + "\n"
            return tao_str, {
                "new_items": new_items_str.strip(),
                "deleted_items": deleted_items_str.strip(),
                "updated_items": updated_items_str.strip()
            }   
    else:
        return "Previous Action did not make any changes", None
    

def compare_observations(obs1, obs2):
    mnkrs = Munkres()
    matrix = []

    if len(obs1.items()) == 0 or len(obs2.items()) == 0:
        return [], list(obs2.keys()), []

    for i, (key1, item1) in enumerate(obs1.items()):
        row = []
        for j, (key2, item2) in enumerate(obs2.items()):
            role1, name1 = item1['role'], item1['content']
            role2, name2 = item2['role'], item2['content']

            string_match_score = -int(name1 == name2) * 10
            location_score = abs(int(key2) - int(key1)) / 10
            role_score = -int(role1 == role2) / 2 

            final_score = string_match_score + location_score + role_score
            row.append(final_score)
        matrix.append(row)

    best_idxs = mnkrs.compute(matrix)

    matched_keys = set()
    updated_keys = []
    for row, col in best_idxs:
        key1 = list(obs1.keys())[row]
        key2 = list(obs2.keys())[col]
        matched_keys.add(key2)
        if obs1[key1] != obs2[key2]:
            updated_keys.append(key2)

    new_items = [ obs2[key] for key in obs2.keys() if key not in matched_keys]
    deleted_items = [obs1[key] for key in obs1.keys() if key not in [list(obs2.keys())[col] for _, col in best_idxs]]
    updated_items = [obs2[key] for key in updated_keys]

    return new_items, deleted_items, updated_items


def compare_observations_torch(obs1, obs2):
    if len(obs1.items()) == 0 or len(obs2.items()) == 0:
        return [], list(obs2.keys()), []

    # Convert the dictionary items to lists for easier indexing
    obs1_items = list(obs1.items())
    obs2_items = list(obs2.items())

    # Prepare the cost matrix for the Hungarian algorithm
    cost_matrix = []
    for i, (key1, item1) in enumerate(obs1.items()):
        row = []
        for j, (key2, item2) in enumerate(obs2.items()):
            role1, name1 = item1['role'], item1['content']
            role2, name2 = item2['role'], item2['content']

            string_match_score = -int(name1 == name2) * 100
            location_score = abs(int(key2) - int(key1)) / 1000
            role_score = -int(role1 == role2) / 2 

            final_score = string_match_score + role_score
            row.append(final_score)
        cost_matrix.append(row)
    # Convert the cost matrix to a torch tensor
    cost_matrix_tensor = torch.tensor(cost_matrix, dtype=torch.float32)

    # Log time taken in calculating Hungarian algorithm
    start_time = time.time()
    permutation_matrix = pygm.hungarian(cost_matrix_tensor)
    row_indices, col_indices = torch.where(permutation_matrix == 1)
    end_time = time.time()
    # print(f"Time taken in calculating Hungarian algorithm: {end_time - start_time} seconds")

    matched_keys = set()
    updated_keys = []
    for row, col in zip(row_indices.tolist(), col_indices.tolist()):
        key1 = obs1_items[row][0]
        key2 = obs2_items[col][0]
        matched_keys.add(key2)
        if obs1[key1] != obs2[key2]:
            updated_keys.append(key2)

    new_items = [obs2[key] for key in obs2.keys() if key not in matched_keys]
    deleted_items = [obs1[key] for key in obs1.keys() if key not in [obs2_items[col][0] for row, col in zip(row_indices.tolist(), col_indices.tolist())]]
    updated_items = [obs2[key] for key in updated_keys]

    return new_items, deleted_items, updated_items



if __name__ == "__main__":
    args = parse_args()
    use_torch_version = args.use_torch
    
    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)

    none_count = Value('i', 0)
    none_count_lock = Lock()

    # Function to process a single example
    def process_example(example):
        global none_count
        prev_obs = example['observation']
        next_obs = example['next_state_acctree']
        action = example['current_action']
        if prev_obs is None or next_obs is None or action is None:
            with none_count_lock:
                none_count.value += 1
            return None
        
        tao, update_dict = get_TaO(prev_obs, next_obs, action, use_torch_version)
        
        if update_dict is None:
            return {**example, 'next_state_tao': tao, "new_items": "None", "updated_items": "None", "deleted_items": "None"}
        
        return {**example, 'next_state_tao': tao, "new_items": update_dict['new_items'], "updated_items": update_dict['updated_items'], "deleted_items": update_dict['deleted_items']}

    # Process all examples in the dataset in parallel
    with Pool(processes=cpu_count()) as pool:
        updated_examples = list(tqdm(
            pool.imap(process_example, dataset),
            total=len(dataset),
            desc="Processing examples"
        ))
    
    print(f"None count: {none_count}")

    # Create a new dataset with the updated examples, dropping None values
    filtered_examples = [example for example in updated_examples if example is not None]
    
    output_filename = f"sample_tao_annotated.json"
    with open(os.path.join(args.output_path, output_filename), "w") as f:
        json.dump(filtered_examples, f, indent=4, ensure_ascii=False)