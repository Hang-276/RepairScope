"""Thin extension of the pinned official SimplifiedReActCodeAgent.

Import this module only for live execution: core tooling has no AppWorld import.
The official prompt, code parser, message update and actor step are inherited.
Only the HTTP transport/accounting and finite candidate selection are replaced.
"""
from __future__ import annotations

import time
from pathlib import Path

from appworld_agents.code.common.usage_tracker import Usage, Tokens
from appworld_agents.code.simplified.agent import ExecutionIO
from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent

from .budget import Budget, BudgetExceeded
from .clock import elapsed_time
from .environment import AppWorldFactory
from .io import atomic_json
from .llm import ChatClient, LLMError, message_token_count
from .model_inputs import InputTooLongError
from .policy import (DirectSelector, PolicyError, PolicyStats, ZeroShotEffectPredictor,
                     extract_goals, generate_candidates)
from .runtime import (checkpoint_reasons, count_native_api_calls, trim_actor_messages,
                      validate_live_predictor_metadata)
from .schemas import Candidate
from .selection import select_by_effects, select_by_scalar


class AccountedActor:
    def __init__(self, client, actor_config, seed):
        self.client, self.config, self.seed = client, actor_config, seed

    def generate(self, messages, **kwargs):
        result = self.client.complete(messages, max_tokens=self.config["max_tokens"],
                                      temperature=self.config["temperature"],
                                      seed=self.seed, purpose="actor")
        return {"role": "assistant", "content": result.content,
                "standardized_usage": Usage(tokens=Tokens(
                    input_cache_miss=result.input_tokens, output=result.output_tokens))}

    def log_calls_to(self, **kwargs):
        pass  # Upstream logger already saves executed actor turns; own receipt saves every cost.


class RepairScopeAgent(SimplifiedReActCodeAgent):
    def __init__(self, config, method, split, experiment_name, tokenizer, predictor=None):
        self.config, self.method, self.split = config, method, split
        self.experiment_name, self.predictor = experiment_name, predictor
        self.tokenizer = tokenizer
        self.run_dir = Path(config["output_dir"]) / experiment_name
        self.factory = AppWorldFactory(config["root"], config["seed"])
        if method in {"effect", "scalar"}:
            if predictor is None:
                raise ValueError("Learned online methods require a predictor")
            validate_live_predictor_metadata(predictor.metadata, method)
        super().__init__(
            prompt_file_path=str(Path(config["root"]) / "experiments/prompts/react_code_agent/instructions.txt"),
            model_config={"name": config["actor"]["model"], "client_name": "openai",
                          "cost_per_token": {"input_cache_miss": 0, "input_cache_hit": 0,
                                             "input_cache_write": 0, "output": 0},
                          "max_retries": 1, "use_cache": False},
            appworld_config={"random_seed": config["seed"]},
            logger_config={"verbose": False, "color": False},
            max_steps=config["budget"]["max_steps"], ignore_multiple_calls=True,
            max_prompt_length=None, max_output_length=None, log_lm_calls=False,
            skip_if_finished=False,
        )

    @property
    def trimmed_messages(self):
        actor = self.config["actor"]
        return trim_actor_messages(self.messages, self.num_instruction_messages,
                                   self.tokenizer, actor["context_length"] - actor["max_tokens"],
                                   {"enable_thinking": False})

    def solve_task(self, task_id):
        task_dir = self.run_dir / "tasks" / task_id
        budget_options = {k: v for k, v in self.config["budget"].items() if k != "max_steps"}
        self.budget = Budget(**budget_options)
        actor = self.config["actor"]
        client = ChatClient(actor["base_url"], actor["model"], actor["api_key_env"],
                            tokenizer=self.tokenizer, budget=self.budget,
                            chat_template_kwargs={"enable_thinking": False},
                            context_length=actor["context_length"])
        self.language_model = AccountedActor(client, actor, self.config["seed"])
        stats, goals, steps, checkpoints = PolicyStats(), (), [], []
        termination, eligible = "max_steps", False
        actual_api_calls = 0
        started = elapsed_time()
        with self.factory.create(task_id, self.experiment_name, self.split) as world:
            self.initialize(world)
            initial_messages = [{"role": m["role"], "content": m["content"] or ""}
                                for m in self.messages]
            try:
                if self.method != "baseline":
                    goals = extract_goals(client, world.task.instruction,
                                          max_tokens=self.config["policy"]["goal_tokens"], stats=stats)
                    eligible = len(goals) >= 2 and all(g.kind == "state_write" for g in goals)
                execution_outputs = []
                for step in range(1, self.max_steps + 1):
                    self.step_number = step
                    execution_inputs, usage, status = super().next_execution_inputs_usage_and_status(
                        execution_outputs)
                    if status.failed:
                        termination = "actor_failed"
                        break
                    self.usage_tracker.add(task_id, usage)
                    if len(execution_inputs) != 1:
                        raise ValueError("Pinned simplified ReAct must produce exactly one execution block")
                    code = execution_inputs[0].content
                    if not code.strip():
                        termination = "actor_no_code"
                        break
                    last_output = steps[-1]["output"] if steps else ""
                    reasons = checkpoint_reasons(step, last_output, code,
                                                 self.config["policy"]["checkpoint_interval"])
                    if eligible and reasons:
                        # Includes the official API-use instructions and only actually executed
                        # code/observations. Excludes this step's unexecuted actor draft.
                        # Exclude few-shot demonstration transcripts. The final
                        # instruction message includes actual supervisor identity,
                        # then only this world's actually executed turns follow.
                        history = [{"role": m["role"], "content": m.get("content") or ""}
                                   for m in self.messages[self.num_instruction_messages - 1:-1]]
                        history = self._compact_policy_history(history)
                        try:
                            candidates = generate_candidates(
                                client, world.task.instruction, history, goals, code,
                                max_tokens=self.config["policy"]["candidate_tokens"],
                                max_code_tokens=self.config["policy"]["candidate_code_tokens"],
                                original_max_tokens=actor["max_tokens"], stats=stats)
                        except PolicyError as exc:
                            # Official ReAct executes syntactically bad code to
                            # obtain an observation; preserve that behavior.
                            stats.failed("candidate_invalid_original", exc)
                            candidates = (Candidate("original", code, tuple(g.id for g in goals), 1.0),)
                        selected = candidates[0]
                        probabilities = None
                        predictor_usage = None
                        if len(candidates) > 1:
                            try:
                                args = (world.task.instruction, history, goals, candidates)
                                if self.method == "direct":
                                    selected = DirectSelector(client, stats).select(*args)
                                elif self.method == "rule":
                                    probabilities = ZeroShotEffectPredictor(client, stats).predict(*args)
                                    selected = select_by_effects(candidates, probabilities, len(goals))
                                elif self.method in {"effect", "scalar"}:
                                    estimate = self.predictor.estimate_tokens(*args)
                                    self.budget.check_classifier(estimate)
                                    values = self.predictor.predict(*args)
                                    used = self.predictor.last_usage
                                    self.budget.record_classifier(used["processed_tokens"], used["seconds"])
                                    predictor_usage = dict(used)
                                    if self.method == "effect":
                                        probabilities = values
                                        selected = select_by_effects(candidates, values, len(goals))
                                    else:
                                        selected = select_by_scalar(candidates, values)
                            except (InputTooLongError, PolicyError) as exc:
                                stats.failed("selection_fallback", exc)
                                # Transparent common fallback, never slice goals/code.
                        checkpoint = {"state_id": f"{task_id}:s{len(steps):03d}",
                                      "prefix_length": len(steps), "history": history,
                                      "reasons": reasons, "candidates": [c.to_dict() for c in candidates],
                                      "selected_id": selected.id, "executed": False,
                                      "predictor_usage": predictor_usage}
                        if probabilities is not None:
                            checkpoint["predicted_effects"] = probabilities
                        checkpoints.append(checkpoint)
                        original_code = code
                        code = selected.code
                        # Counterfactual drafts must never masquerade as executed history.
                        if code != original_code:
                            self.messages[-1] = {"role": "assistant", "content": f"```python\n{code}\n```\n\n"}
                    self.budget.ensure_can_execute()
                    output = world.execute(code)
                    steps.append({"code": code, "output": output})
                    if checkpoints and checkpoints[-1]["prefix_length"] == len(steps) - 1:
                        checkpoints[-1]["executed"] = True
                    execution_outputs = [ExecutionIO(content=output)]
                    self.log_usage()
                    if world.task_completed():
                        termination = "declared_complete"  # Native success is evaluated separately.
                        break
            except BudgetExceeded:
                termination = "budget_exhausted"
            except (LLMError, ValueError) as exc:
                termination = type(exc).__name__
                stats.failed("task_runtime", exc)
            finally:
                world.save()
                api_log = Path(world.output_logs_directory) / "api_calls.jsonl"
                actual_api_calls = count_native_api_calls(api_log)
                record = {"schema_version": 1, "task_id": task_id, "split": self.split,
                          "method": self.method, "instruction": world.task.instruction,
                          "goals": [g.to_dict() for g in goals], "eligible": eligible,
                          "initial_messages": initial_messages, "steps": steps,
                          "checkpoints": checkpoints, "seed": self.config["seed"],
                          "termination": termination, "policy": stats.to_dict(),
                          "budget": self.budget.snapshot(), "actual_api_calls": actual_api_calls,
                          "wall_seconds": elapsed_time() - started,
                          "native_usage_scope": "actor_only; task budget includes all auxiliary inference",
                          "evidence_origin": "appworld_runtime"}
                atomic_json(task_dir / "trajectory.json", record)
        self.logger.complete_task()

    def _compact_policy_history(self, history):
        # Leave room for fixed selector prompts, complete task/goals, original code,
        # and the three-candidate JSON envelope. Never cut inside a message.
        while len(history) > 1 and message_token_count(history, self.tokenizer) > 2600:
            history = history[1:]
        return history
