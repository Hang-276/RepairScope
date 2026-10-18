import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--run_index", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    return parser.parse_args()


class TrainDataConstructor():
    def __init__(self, trajectory_path, output_path, run, start, end):
        self.all_data = []
        self.trajectory_path = trajectory_path
        self.output_path = output_path
        self.run_id = run - 1
        self.start = start
        self.end = end

    def construct(self):
        for i in range(self.start, self.end + 1):
            file_name = f"trajectory_{i}.json"
            file_path = os.path.join(self.trajectory_path, file_name)
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        for sample in data:
                            temp_data = {}
                            temp_data["task_id"] = sample["task_id"]
                            temp_data["run_id"] = self.run_id
                            temp_data["step_idx"] = sample["step_idx"]
                            temp_data["url"] = sample["state_info"]["url"]
                            temp_data["objective"] = sample["intent"]
                            temp_data["observation"] = sample["state_info"]["observation"]["text"] # AccTree
                            temp_data["previous_actions"] = sample["previous_actions"]
                            temp_data["current_action"] = sample["current_action_str"]
                            temp_data["next_state_acctree"] = sample["next_state_acctree"]["observation"]["text"] # AccTree
                            self.all_data.append(temp_data)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to decode JSON from the file {file_name}")
            else:
                print(f"Warning: File not found: {file_name}")
        return self.all_data
            
    def save(self, data):
        save_path = self.output_path
        file_name = "next_state_acctree.json"
        save_path_ = os.path.join(save_path, file_name)
        with open(save_path_, "w", encoding='UTF-8') as f:
            json.dump({"data": data}, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    output_path = args.output_path
    run_index = args.run_index
    start = args.start
    end = args.end

    total_data = []
    for run in range(1, run_index + 1):
        trajectory_path = args.trajectory_path
        data_constructor = TrainDataConstructor(trajectory_path, output_path, run, start, end)
        total_data.extend(data_constructor.construct())
    data_constructor.save(total_data)