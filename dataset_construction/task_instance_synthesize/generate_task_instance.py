import argparse
import asyncio
import json
import numpy as np
import pandas as pd
import sys
import os
import random

from tqdm.asyncio import tqdm as tqdm_async
from datasets import load_dataset
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from tqdm import tqdm
from utils import generate_concurrently, async_generate
from nr_openai_observability import monitor


parser = argparse.ArgumentParser()
parser.add_argument("-nd", "--num_demonstrations", type=int, default=10)
parser.add_argument("-ds","--data_synthesize_strategy", type=str, choices=[
    "from_existing_template",
    "from_generated_template"
], default="from_existing_template")
parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
parser.add_argument("-nt","--num_template", type=int, default=None, help="you can set num_template as a small number to test if your code works")
parser.add_argument("--min_num_of_data", type=int, default=10000) # 10K

INSTANTIATION_GENERATION = "from_existing_template"
TEMPLATE_GENERATION = "from_generated_template"
generation_types = [INSTANTIATION_GENERATION, TEMPLATE_GENERATION]


def load_website_description():
    description_path = "task_instance_synthesize/webarena_description.json"
    with open(description_path, "r") as f:
        website_descriptions = json.load(f)
    
    return website_descriptions


def load_prompt(args):
    prompt_path_dict = {
        INSTANTIATION_GENERATION: "task_instance_synthesize/prompt_for_instantiation_generation.txt",
        TEMPLATE_GENERATION: "task_instance_synthesize/prompt_for_template_generation.txt"
    }
    prompt_path = prompt_path_dict[args.data_synthesize_strategy]
    with open(prompt_path, "r") as f:
        prompt = f.read()
    print(f"Prompt for {args.data_synthesize_strategy} is loaded.") 
    return prompt


def load_test_data_by_website():
    test_data_path = "task_instance_synthesize/dataset/sample_data_10.json"
    with open(test_data_path, "r") as f:
        test_data = json.load(f)
    # save test data by website name
    test_data_by_website_name = {}
    muilti_website_tasks = []
    for td in test_data:
        cur_website_name = td["sites"] # website names are stored in list type
        if len(cur_website_name) >= 2: # IDK if there are instances that are assigned to multiple webistes
            muilti_website_tasks.append(td)
            continue
        cur_website_name = cur_website_name[0] # get the first element of the list ex. ... "sites": ["shopping_admin"] ...
        if cur_website_name not in test_data_by_website_name:
            test_data_by_website_name[cur_website_name] = [td]
        else:
            test_data_by_website_name[cur_website_name].append(td)
    
    print(f"num multi-website tasks: {len(muilti_website_tasks)}") 
    return test_data_by_website_name 


def collect_template_by_website(test_data_by_website_name):
    template_by_website = {wn:{} for wn in test_data_by_website_name.keys()}
    template_to_full_data = {}

    for wn, data_of_website in test_data_by_website_name.items():
        for d in data_of_website:
            cur_template = d['intent_template']
            cur_instantiation = d['instantiation_dict']
            if cur_template not in template_by_website[wn]:
                template_by_website[wn][cur_template] = [cur_instantiation]
                template_to_full_data[cur_template] = d
            else:
                template_by_website[wn][cur_template].append(cur_instantiation)
    
    return template_by_website, template_to_full_data


def prepare_model_input_for_instatiation_generation(args, prompt, template_by_website):
    all_model_inputs = []
    all_templates_flatten = []
    for ws, template_dict in template_by_website.items():

        for template, example_instatiation in template_dict.items():
            model_input = prompt.format(
                template=template,
                example_instantiation= example_instatiation
            )
            all_templates_flatten.append(template)
            all_model_inputs.append(model_input)
        
    return all_model_inputs, all_templates_flatten


async def main():
    args = parser.parse_args()
    test_data_by_website_name = load_test_data_by_website()
    prompt = load_prompt(args)

    for ws_name, td in test_data_by_website_name.items():
        print(f"{ws_name}: {len(td)}")

    template_by_website, template_to_full_data = collect_template_by_website(test_data_by_website_name)
    
    with open("task_instance_synthesize/template_by_website.json", "w") as f:
        json.dump(template_by_website, f, indent=4)

    if args.model_name == "gpt-4":
        consent_to_use_gpt4 = input("You have chosen GPT-4 for sythesizing the data. Are you ready to be a GPT-broke? (y/n)")
        if consent_to_use_gpt4 != "y":
            exit()

    llm = ChatOpenAI(
        model_name=args.model_name,
        temperature = 1.0,
        max_retries=100,
        stop=["[Example"],
        max_tokens=3000
    )
    
    if args.data_synthesize_strategy == TEMPLATE_GENERATION:
        pass
    elif args.data_synthesize_strategy == INSTANTIATION_GENERATION:
        all_model_inputs, all_templates_flatten = prepare_model_input_for_instatiation_generation(args, prompt, template_by_website)
        if args.num_template:
            all_model_inputs = all_model_inputs[:args.num_template]
            all_templates_flatten = all_templates_flatten[:args.num_template]
    
    model_outputs = await generate_concurrently(llm, all_model_inputs, None)
    with open("task_instance_synthesize/input_output_synthesized.json", "w") as f:
        list_to_be_saved = [
            {"input":all_model_inputs[i].split("\n"), "output": model_outputs[i].split("\n")}
            for i in range(len(model_outputs))
        ]
        json.dump(list_to_be_saved, f, indent=4)

    matched_data = [
        {k:v for k,v in template_to_full_data[template].items()} for template in all_templates_flatten
    ]
    for i in range(len(model_outputs)):
        if "'" in model_outputs[i]:
            model_outputs[i] = model_outputs[i].replace("'", '"')

        try:
            parsed_template = json.loads(model_outputs[i])
        except:
            print(f"Could not parse template. Generate output: {model_outputs[i]}")
            parsed_template = []

        if type(parsed_template) == dict:
            parsed_template = [parsed_template]
        
        # print(f"{i}-th output: {parsed_template}")
        matched_data[i]['instantiation_dict'] = parsed_template

    synthesized_data = []
    for md in matched_data:
        for i in range(len(md['instantiation_dict'])):
            print(json.dumps(md, indent=4))
            tmp_dict = {k:v for k,v in md.items()}
            tmp_dict['instantiation_dict'] = tmp_dict['instantiation_dict'][i]
            synthesized_data.append(tmp_dict)

    dir_path = f"task_instance_synthesize/synthesized_instances/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for i in range(len(synthesized_data)):
        with open(f"task_instance_synthesize/synthesized_instances/{i}.json", "w") as f:
            json.dump(synthesized_data[i], f, indent=4)
            
    
if __name__ == "__main__":
    asyncio.run(main())