import json
import os

data_dir = "task_instance_synthesize/synthesized_instances"
formatted_data_dir = "task_instance_synthesize/synthesized_instances"
files = os.listdir(data_dir)

formatted_data = []
for fn in files:
    print(fn)
    with open(os.path.join(data_dir, fn), "r") as f:
        cur_data = json.load(f)
    formatted_intent = cur_data["intent_template"].replace("{{", "{").replace("}}","}").format(**cur_data['instantiation_dict'])
    
    cur_data["prev_intent"] = cur_data["intent"]
    cur_data["original_task_id"] = cur_data["task_id"]
    cur_data["task_id"] = int(fn.split(".")[0])
    cur_data['intent'] = formatted_intent

    formatted_data.append(cur_data)

    with open(os.path.join(formatted_data_dir, fn), "w") as f:
        json.dump(cur_data, f, indent=4)

    
        