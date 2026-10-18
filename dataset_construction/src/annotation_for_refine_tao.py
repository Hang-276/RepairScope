import os
import argparse
import asyncio
import json
import time

from copy import deepcopy
from tqdm.asyncio import tqdm_asyncio
from langchain_openai import ChatOpenAI
from datasets import load_dataset
from langchain_core.messages import SystemMessage
from langchain.callbacks import get_openai_callback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--prompt_path", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use for annotation")
    parser.add_argument("--batch_size", type=int, default=200, help="Number of concurrent async tasks")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def generate_input(instance):
    with open(args.prompt_path, "r") as f:
        prompt = json.load(f)
    system_prompt = prompt["intro"]

    system_prompt = system_prompt.format(new_items=instance['new_items'], updated_items=instance['updated_items'], deleted_items=instance['deleted_items'])

    # system prompt
    messages = [SystemMessage(content=system_prompt)]
    return messages


async def process_item(instance, llm):
    # save original instance
    result = deepcopy(instance)
    
    # generate
    final_input = generate_input(instance)
    response = await llm.agenerate([final_input])
    result['refined_tao'] = response.generations[0][0].text
    return result


async def process_dataset(dataset, llm, batch_size, args):
    results = []
    total_cost = 0
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:min(i+batch_size, len(dataset))]
        print(f"Processing batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1} (size: {len(batch)})")
        with get_openai_callback() as cb:
            tasks = [process_item(instance, llm) for instance in batch]
            batch_results = await tqdm_asyncio.gather(*tasks, desc=f"Processing batch {i//batch_size + 1}")
            results.extend(batch_results)
            
            if args.model == "gpt-4o-mini":
                batch_cost = (cb.prompt_tokens*0.00015)/1000+(cb.completion_tokens*0.0006)/1000
            else:
                batch_cost = cb.total_cost
            
            total_cost += batch_cost
            print(f"Batch Cost: ${batch_cost:.6f}")
            print(f"Total Cost so far: ${total_cost:.6f}")
            print(f"Total tokens: {cb.total_tokens}")
            print(f"Prompt tokens: {cb.prompt_tokens}")
            print(f"Completion tokens: {cb.completion_tokens}")
    
    return results, total_cost


async def annotate_dataset_async(dataset, llm, batch_size, args):
    return await process_dataset(dataset, llm, batch_size, args)


async def annotate_dataset(args):
    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)

    # load model
    llm = ChatOpenAI(model=args.model, api_key=args.api_key)

    # annotate dataset
    start_time = time.time()
    results, total_cost = await annotate_dataset_async(dataset, llm, args.batch_size, args)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Total Cost (USD): ${total_cost:.6f}")

    output_filename = f"sample_refined_tao_annotated.json"
    
    with open(os.path.join(args.output_path, output_filename), "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(annotate_dataset(args))
