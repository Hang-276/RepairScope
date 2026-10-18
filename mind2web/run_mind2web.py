import os
import argparse
from tqdm import tqdm
from mind2web.memory import eval_sample
from mind2web.utils.data import load_json, add_scores

import logging
logger = logging.getLogger("atm")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


from multiprocessing import Process


def run(args: argparse.Namespace, examples_: list[dict]) -> None:
    examples = [s for s in examples_ if s["website"] == args.website]
    print(f"Filtering down to #{len(examples)} examples on website [{args.website}]")
    examples = add_scores(examples) # add prediction scores and ranks to elements

    if args.end_idx is None:
        args.end_idx = len(examples)
    for i in tqdm(range(args.start_idx, args.end_idx)):

        args.domain = examples[i]["domain"]
        args.subdomain = examples[i]["subdomain"]

        if args.mode == "memory":
            eval_sample(i, args, examples[i])
        elif args.mode == "action":
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported workflow format: {args.workflow_format}")


def main():
    for b in ["test_task", "test_website", "test_domain"]:
        args.benchmark = b

        examples_ = load_json(args.data_dir, args.benchmark)
        websites = [*set([s["website"] for s in examples_])]

        ps = []

        for w in tqdm(websites):
            args.website = w
            args.workflow_path = f"mind2web/workflow/{w}.txt"

            p = Process(target=lambda: run(args, examples_))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="mind2web/data")
    parser.add_argument("--benchmark", type=str, default="test_task",
        choices=["test_task", "test_website", "test_domain", "train"])
    parser.add_argument("--memory_path", type=str, default="mind2web/data/memory")
    parser.add_argument("--log_dir", type=str, default="mind2web/results")

    # model
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=1.0)

    # env context
    parser.add_argument("--previous_top_k_elements", type=int, default=3)
    parser.add_argument("--top_k_elements", type=int, default=5)
    parser.add_argument("--retrieve_top_k", type=int, default=1)

    # workflow
    parser.add_argument("--website", type=str, default="")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--subdomain", type=str, default=None)
    parser.add_argument("--workflow_path", type=str, default="mind2web/workflow/asdf")
    parser.add_argument("--suffix", type=str, default="workflow")

    # ablation
    parser.add_argument("--mode", type=str, default="memory", choices=["memory", "action"])
    parser.add_argument("--start_idx", type=int, default=0, help="Select example index.")
    parser.add_argument("--end_idx", type=int, default=None, help="Select example index.")

    # world model & value function
    parser.add_argument("--branching_factor", type=int, default=3)
    parser.add_argument("--vf_budget", type=int, default=20)
    parser.add_argument("--world_model_training", action="store_true")
    parser.add_argument("--world_model_name", type=str, default=None)
    parser.add_argument("--world_model_url", type=str, default=None)
    parser.add_argument("--value_model_training", action="store_true")
    parser.add_argument("--value_model_name", type=str, default=None)
    parser.add_argument("--value_model_url", type=str, default=None)

    args = parser.parse_args()

    # sanity check
    if not os.path.exists(args.workflow_path): open(args.workflow_path, 'w').close()
    if args.retrieve_top_k != 1: print(f"Suggest set `retrieve_top_k` to 1, currently as {args.retrieve_top_k}")

    if args.world_model_training:
        assert args.world_model_name is not None and args.world_model_url is not None
    if args.value_model_training:
        assert args.value_model_name is not None and args.value_model_url is not None

    main()
