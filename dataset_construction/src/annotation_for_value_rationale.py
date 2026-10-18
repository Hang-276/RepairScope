import os
import argparse
import asyncio
import json
import re
import base64
import time

from PIL import Image
from io import BytesIO
from copy import deepcopy
from tqdm.asyncio import tqdm_asyncio
from langchain_openai import ChatOpenAI
from datasets import load_dataset, Dataset
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback
import tiktoken


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--prompt_path", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use for annotation")
    parser.add_argument("--batch_size", type=int, default=300, help="Number of concurrent async tasks")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def generate_input(item):
    with open(args.prompt_path, "r") as f:
        prompt = json.load(f)
    system_template = prompt["intro"]
    template = prompt["template"]

    # system prompt
    messages = [SystemMessage(content=system_template)]
    
    user_content = template.format(
        objective=item["confirmed_task"],
        prev_action=item["previous_actions"],
        cur_observation=item["cleaned_accessibility_tree"],
        cur_action=item["action"],
        next_state_prediction=item["next_state_description_with_tao"],
        actual_score=item["value_score"]
    )
    messages.append(HumanMessage(content=user_content))

    return messages


def parsing(raw_prediction):
    rationale_pattern = r"\[Rationale\](.*?)\[Score\]"
    score_pattern = r"\[Score\](.*)"

    rationale_match = re.search(rationale_pattern, raw_prediction, re.DOTALL)
    score_match = re.search(score_pattern, raw_prediction, re.DOTALL)

    rationale = rationale_match.group(1).strip() if rationale_match else ""
    score = score_match.group(1).strip() if score_match else ""

    return rationale, score


async def process_item(instance, llm):
    # save original instance
    result = deepcopy(instance)
    
    # generate
    final_input = generate_input(instance)
    response = await llm.agenerate([final_input])
    result['value_raw_prediction'] = response.generations[0][0].text
    rationale, score = parsing(result['value_raw_prediction'])
    result['value_rationale'] = rationale
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
    if args.debug:
        dataset = dataset[:10]
    else:
        dataset = dataset

    # load model
    llm = ChatOpenAI(model=args.model, api_key=args.api_key)

    # annotate dataset
    start_time = time.time()
    results, total_cost = await annotate_dataset_async(dataset, llm, args.batch_size, args)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Total Cost (USD): ${total_cost:.6f}")

    # save results
    output_filename = f"value_rationale_annotated.json"
    with open(os.path.join(args.output_path, output_filename), "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(annotate_dataset(args))
