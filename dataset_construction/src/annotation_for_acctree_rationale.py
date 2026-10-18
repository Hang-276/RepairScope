import os
import argparse
import asyncio
import json
import re
import time

from PIL import Image
from io import BytesIO
from copy import deepcopy
from tqdm.asyncio import tqdm_asyncio
from langchain_openai import ChatOpenAI
from datasets import load_dataset
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback
from datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--prompt_path", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use for annotation")
    parser.add_argument("--batch_size", type=int, default=300, help="Number of concurrent async tasks")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def generate_input(prompt, item):
    system_template = prompt["instruction"]
    examples = prompt["demonstrations"]
    template = prompt["template"]

    # system prompt
    messages = [SystemMessage(content=system_template)]

    example_template = "URL: {url} \nOBJECTIVE: {objective} \nPREVIOUS ACTION: {prev_action} \
                        \nCURRENT OBSERVATION: {cur_observation} \nCURRENT ACTION: {cur_action} \
                        \nACTUAL NEXT STATE OBSERVATION: {actual_next_state_observation}"
    for example in examples:
        example_content = example_template.format(
            url=example["url"],
            objective=example["objective"],
            prev_action=example["prev_action"],
            cur_observation=example["cur_observation"],
            cur_action=example["cur_action"],
            actual_next_state_observation=example["actual_next_state_observation"],
        )
        messages.extend([
            HumanMessage(content=example_content),
            AIMessage(content=example["rationale"]),
        ])

    user_content = template.format(
        url=item["url"],
        objective=item["objective"],
        prev_action=item["previous_actions"],
        cur_observation=item["observation"],
        cur_action=item["current_action"],
        actual_next_state_observation=item["next_state_acctree"]
    )

    messages.append(HumanMessage(content=user_content))
    return messages


async def process_item(instance, prompt_template, llm):
    # save original instance
    result = deepcopy(instance)
    
    # generate
    final_input = generate_input(prompt_template, instance)
    response = await llm.agenerate([final_input])
    result['next_state_rationale'] = response.generations[0][0].text

    return result


async def process_dataset(dataset, prompt_template, llm, batch_size, args):
    results = []
    total_cost = 0
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:min(i+batch_size, len(dataset))]
        print(f"Processing batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1} (size: {len(batch)})")
        with get_openai_callback() as cb:
            tasks = [process_item(instance, prompt_template, llm) for instance in batch]
            batch_results = await tqdm_asyncio.gather(*tasks, desc=f"Processing batch {i//batch_size + 1}")
            results.extend(batch_results)
            
            if args.model == "gpt-4o-mini":
                batch_cost = (cb.prompt_tokens*0.00015)/1000+(cb.completion_tokens*0.0006)/1000
            elif args.model == "gpt-4o-2024-08-06":
                batch_cost = (cb.prompt_tokens*0.0025)/1000+(cb.completion_tokens*0.01)/1000
            else:
                batch_cost = cb.total_cost
            
            total_cost += batch_cost
            print(f"Batch Cost: ${batch_cost:.6f}")
            print(f"Total Cost so far: ${total_cost:.6f}")
            print(f"Total tokens: {cb.total_tokens}")
            print(f"Prompt tokens: {cb.prompt_tokens}")
            print(f"Completion tokens: {cb.completion_tokens}")
    
    return results, total_cost


async def annotate_dataset_async(dataset, prompt_template, llm, batch_size, args):
    return await process_dataset(dataset, prompt_template, llm, batch_size, args)


async def annotate_dataset(args):
    prompt_path = args.prompt_path
    with open(prompt_path, "r") as f:
        prompt_template = json.load(f)
    
    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)

    # load model
    llm = ChatOpenAI(model=args.model, api_key=args.api_key)

    # annotate dataset
    start_time = time.time()
    results, total_cost = await annotate_dataset_async(dataset, prompt_template, llm, args.batch_size,  args)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Total Cost (USD): ${total_cost:.6f}")

    output_filename = f"sample_acctree_cot_annotated_{args.model}.json"
    with open(os.path.join(args.output_path, output_filename), "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(annotate_dataset(args))
