"""Train/dev-only paired counterfactual collection in fresh processes.

Trajectory JSON contains task_id, split, instruction, goals, steps [{code,output}]
and checkpoints [{state_id,prefix_length,history,candidates}]. A checkpoint's
history is the *visible* pre-candidate conversation. Original failed code blocks
are replayed unchanged: a failure can leave writes and Python variables behind.
"""
from __future__ import annotations

import importlib
import json
import multiprocessing as mp
from pathlib import Path
import traceback
from typing import Any, Callable, Iterable
import uuid

from .environment import AppWorldFactory, fingerprint_world, stable_digest, tracker_status
from .io import atomic_json, write_jsonl
from .schemas import Candidate, Example, Goal, IGNORE_LABEL, ensure_research_split

MAPPING_SCHEMA_VERSION = 1


def _trajectory_paths(location: str | Path) -> list[Path]:
    location = Path(location)
    if location.is_file():
        return [location]
    if (location / "tasks").is_dir():
        return sorted((location / "tasks").glob("*/trajectory.json"))
    nested = sorted(location.glob("*/trajectory.json"))
    if nested:
        return nested
    # Flat task-ID JSON files remain supported. Run summaries, mapping files and
    # collection receipts in the same folder are not trajectory inputs.
    required = {"task_id", "split", "instruction", "goals", "steps", "checkpoints"}
    result = []
    for path in sorted(location.glob("*.json")):
        if path.name == "run.json" or path.name.endswith(".receipt.json"):
            continue
        value = json.loads(path.read_text())
        if isinstance(value, dict) and required.issubset(value):
            result.append(path)
    return result


def _goals(instruction: str, values: Iterable[Any]) -> tuple[Goal, ...]:
    goals = tuple(g if isinstance(g, Goal) else Goal.from_dict(g) for g in values)
    if not goals or len({g.id for g in goals}) != len(goals):
        raise ValueError("Nonempty, unique instruction-level goals required")
    if any(g.evidence_span not in instruction for g in goals):
        raise ValueError("Every goal needs exact original-instruction evidence")
    return goals


def create_mapping(task_id: str, split: str, instruction: str, goals: Iterable[Any],
                   requirements: Iterable[str], mapper: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    """Map train/dev checker requirements to existing high-level goals.

    mapper is invoked with instruction=..., goals=[public Goal dicts], and
    requirements=[strings]. It must return goal_requirements (or that mapping
    directly), with each value {requirements:[...],complete:bool}. Multiple
    requirements can support one goal; unmatched/incomplete goals remain masked.
    No evaluator source code, private data or failure trace is supplied.
    """
    ensure_research_split(split)
    goals = _goals(instruction, goals)
    requirements = sorted(set(map(str, requirements)))
    if not requirements:
        raise ValueError("No evaluator requirements available")
    answer = mapper(instruction=instruction, goals=[g.to_dict() for g in goals], requirements=requirements)
    groups = answer.get("goal_requirements", answer)
    normalized = _validate_groups(groups, goals, set(requirements), allow_lists=False)
    return {"task_id": task_id, "split": split,
            "instruction_sha256": stable_digest(instruction),
            "goals": [g.to_dict() for g in goals], "goal_requirements": normalized,
            "audit": {"source": "requirement_mapper", "requirements_sha256": stable_digest(requirements),
                      "requirement_count": len(requirements),
                      "note": "Completeness is a mapping claim; audit train/dev mapping accuracy before main training."}}


def _validate_groups(groups: Any, goals: tuple[Goal, ...], known: set[str] | None = None,
                     allow_lists: bool = True) -> dict[str, dict[str, Any]]:
    if not isinstance(groups, dict) or set(groups) - {g.id for g in goals}:
        raise ValueError("Mapping refers to unknown goal IDs")
    result = {}
    for goal in goals:
        entry = groups.get(goal.id, {"requirements": [], "complete": False})
        if isinstance(entry, list) and allow_lists:
            # An explicit audited mapping file may use a compact list spelling.
            entry = {"requirements": entry, "complete": True}
        if not isinstance(entry, dict) or not isinstance(entry.get("complete"), bool):
            raise ValueError("Mapping entry requires requirements and explicit complete boolean")
        reqs = entry.get("requirements")
        if not isinstance(reqs, list) or any(not isinstance(x, str) or not x for x in reqs):
            raise ValueError("Requirements must be a list of exact checker strings")
        if len(reqs) != len(set(reqs)) or (known is not None and set(reqs) - known):
            raise ValueError("Duplicate or unknown evaluator requirements in mapping")
        result[goal.id] = {"requirements": reqs, "complete": bool(entry["complete"] and reqs)}
    # One joint/global checker cannot identify each component goal's truth.
    # Shared checks therefore cannot justify independent per-goal labels even
    # when the mapper has optimistically declared both groups complete.
    owners: dict[str, list[str]] = {}
    for goal_id, entry in result.items():
        for requirement in entry["requirements"]:
            owners.setdefault(requirement, []).append(goal_id)
    for goal_ids in owners.values():
        if len(goal_ids) > 1:
            for goal_id in goal_ids:
                result[goal_id]["complete"] = False
    return result


def mapped_goal_status(goals: tuple[Goal, ...], groups: dict[str, Any],
                       statuses: dict[str, bool]) -> list[bool | None]:
    output = []
    for goal in goals:
        group = groups[goal.id]
        reqs = group["requirements"]
        if goal.kind != "state_write" or not group["complete"] or not reqs or any(r not in statuses for r in reqs):
            output.append(None)
        else:
            output.append(all(statuses[r] for r in reqs))
    return output


def transition_labels(before: list[bool | None], after: list[bool | None]) -> tuple[int, ...]:
    if len(before) != len(after):
        raise ValueError("Goal status lengths differ")
    return tuple(IGNORE_LABEL if b is None or a is None else int(b) * 2 + int(a)
                 for b, a in zip(before, after))


def _history(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("Checkpoint visible history must be a list")
    result = []
    for message in value:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("History may contain only visible role/content, never evaluator metadata")
        if message["role"] not in {"system", "user", "assistant", "tool"} or not isinstance(message["content"], str):
            raise ValueError("Invalid visible message")
        result.append(dict(message))
    return result


def _resolve_factory(spec: str | None, root: str, seed: int) -> Any:
    if spec is None:
        return AppWorldFactory(root, seed)
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)(root, seed)


def _replay_worker(connection: Any, request: dict[str, Any]) -> None:
    """One worker creates exactly one world and never imports test ground truth."""
    try:
        split = request["split"]
        ensure_research_split(split)
        factory = _resolve_factory(request.get("factory_spec"), request["root"], request["seed"])
        with factory.create(request["task_id"], request["experiment_name"], split,
                            full_ground_truth=True) as world:
            observations = []
            for step in request["steps"]:
                # Do not skip execution-failure observations or rollback writes.
                output = world.execute(step["code"])
                if not isinstance(output, str):
                    raise ValueError("Environment execute must return observation text")
                observations.append(output)
            before_fingerprint = fingerprint_world(world)
            # evaluate_task loads the task's output DB path. Empty prefixes and
            # syntax failures may not have produced that path yet.
            world.save()
            before_status = tracker_status(world.evaluate())
            candidate = request.get("candidate")
            candidate_output = None
            after_status = before_status
            after_fingerprint = before_fingerprint
            if candidate is not None:
                candidate_output = world.execute(candidate["code"])
                after_fingerprint = fingerprint_world(world)
                world.save()
                after_status = tracker_status(world.evaluate())
            result = {"ok": True, "before_fingerprint": before_fingerprint,
                      "after_fingerprint": after_fingerprint, "observations": observations,
                      "observations_sha256": stable_digest(observations),
                      "before_status": before_status, "after_status": after_status,
                      "candidate_output": candidate_output}
        connection.send(result)
    except BaseException as error:
        connection.send({"ok": False, "error": f"{type(error).__name__}: {error}",
                         "traceback": traceback.format_exc()})
    finally:
        connection.close()


def replay_in_fresh_process(request: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
    """spawn prevents inherited AppWorld caches, open SQLite handles and shells."""
    context = mp.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_replay_worker, args=(writer, request))
    process.start()
    writer.close()
    try:
        if not reader.poll(timeout):
            process.terminate()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join()
            return {"ok": False, "error": f"Replay process timed out after {timeout}s"}
        try:
            result = reader.recv()
        except EOFError:
            result = {"ok": False, "error": "Replay process exited without a receipt"}
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join()
            return {"ok": False, "error": "Replay process failed to exit cleanly"}
        if process.exitcode != 0:
            return {"ok": False, "error": f"Replay process exit code {process.exitcode}"}
        return result
    finally:
        reader.close()


def build_mappings_from_trajectories(trajectory_dir: str | Path, output: str | Path,
                                    root: str | Path, mapper: Callable[..., dict[str, Any]],
                                    seed: int = 100, timeout: float = 300,
                                    factory_spec: str | None = None) -> dict[str, Any]:
    """Read initial train/dev requirement strings in isolated labeler processes.

    The callback receives original goals/instruction + requirement strings only.
    The resulting private file must not be used as online agent input. Mapping
    completeness is recorded, not certified by this mechanically valid builder.
    """
    paths = _trajectory_paths(trajectory_dir)
    if not paths:
        raise ValueError("No trajectory files found")
    tasks: dict[str, Any] = {}
    skipped_no_checkpoints = 0
    for path in paths:
        trajectory = json.loads(path.read_text())
        split, task_id = trajectory["split"], trajectory["task_id"]
        ensure_research_split(split)
        if not trajectory.get("checkpoints"):
            skipped_no_checkpoints += 1
            continue
        goals = _goals(trajectory["instruction"], trajectory["goals"])
        if task_id in tasks:
            previous = tasks[task_id]
            if (previous["instruction_sha256"] != stable_digest(trajectory["instruction"]) or
                previous["goals"] != [g.to_dict() for g in goals] or previous["split"] != split):
                raise ValueError(f"Conflicting trajectories for {task_id}")
            continue
        result = replay_in_fresh_process({"root": str(Path(root).resolve()), "seed": seed,
            "task_id": task_id, "split": split, "steps": [], "factory_spec": factory_spec,
            "experiment_name": "b2_mapping_" + uuid.uuid4().hex[:16]}, timeout)
        if not result["ok"]:
            raise RuntimeError(f"Could not obtain train/dev requirements for {task_id}: {result['error']}")
        tasks[task_id] = create_mapping(task_id, split, trajectory["instruction"], goals,
            result["before_status"].keys(), mapper)
    document = {"schema_version": MAPPING_SCHEMA_VERSION, "tasks": tasks,
                "private_labeler_only": True,
                "evidence_origin": "AppWorld" if factory_spec is None else "synthetic_fixture"}
    atomic_json(output, document)
    return {"tasks": len(tasks), "skipped_no_checkpoints": skipped_no_checkpoints, "output": str(output),
            "mapping_sha256": stable_digest(document),
            "evidence_origin": document["evidence_origin"]}


def collect_from_trajectories(trajectory_dir: str | Path, mapping_file: str | Path,
                              output: str | Path, root: str | Path, seed: int = 100,
                              timeout: float = 300, experiment_prefix: str = "b2_collect",
                              factory_spec: str | None = None) -> dict[str, Any]:
    """Collect auditable Examples and sibling .receipt.json/.states.jsonl.

    A failure rejects the affected state/candidate and remains in receipts.
    Output contains no checker requirements, checker status vector, failure
    trace, oracle label or unexecuted branch observation in model history.
    """
    trajectory_dir, output = Path(trajectory_dir), Path(output)
    mappings = json.loads(Path(mapping_file).read_text())
    if mappings.get("schema_version") != MAPPING_SCHEMA_VERSION or not isinstance(mappings.get("tasks"), dict):
        raise ValueError("Mapping file needs schema_version=1 and tasks mapping")
    paths = _trajectory_paths(trajectory_dir)
    if not paths:
        raise ValueError("No trajectory JSON files found")
    examples: list[dict[str, Any]] = []
    state_receipts = []
    rejected = 0
    seen_states: set[tuple[str, str]] = set()
    source_files = []
    skipped_no_checkpoints = 0
    for path in paths:
        raw = path.read_bytes()
        trajectory = json.loads(raw)
        split, task_id = trajectory["split"], trajectory["task_id"]
        ensure_research_split(split)
        source = "synthetic_fixture" if factory_spec is not None else trajectory.get("source", "natural")
        source_files.append({"name": path.name, "sha256": stable_digest(trajectory)})
        if not trajectory.get("checkpoints"):
            skipped_no_checkpoints += 1
            continue
        instruction = trajectory["instruction"]
        goals = _goals(instruction, trajectory["goals"])
        mapping = mappings["tasks"].get(task_id)
        if mapping is None or mapping.get("split") != split:
            raise ValueError(f"Missing/mismatched train/dev mapping for {task_id}")
        if mapping.get("instruction_sha256") != stable_digest(instruction) or mapping.get("goals") != [g.to_dict() for g in goals]:
            raise ValueError(f"Mapping instruction/goals mismatch for {task_id}")
        groups = _validate_groups(mapping["goal_requirements"], goals)
        steps = trajectory["steps"]
        if any(not isinstance(step.get("code"), str) or not isinstance(step.get("output"), str) for step in steps):
            raise ValueError("Every executed step needs original code and output")
        for checkpoint in trajectory["checkpoints"]:
            state_id = str(checkpoint["state_id"])
            state_key = task_id, state_id
            if state_key in seen_states:
                raise ValueError(f"Duplicate task/state: {state_key}")
            seen_states.add(state_key)
            prefix = checkpoint["prefix_length"]
            if not isinstance(prefix, int) or not 0 <= prefix <= len(steps):
                raise ValueError("Checkpoint prefix_length out of bounds")
            history = _history(checkpoint["history"])
            candidates = [Candidate.from_dict(c) for c in checkpoint["candidates"]]
            if not candidates or len({c.id for c in candidates}) != len(candidates):
                raise ValueError("Checkpoint requires nonempty unique candidates")
            if any(set(c.target_goal_ids) - {g.id for g in goals} for c in candidates):
                raise ValueError("Candidate refers to unknown goal")
            pool_max_cost = max(1.0, max(c.estimated_api_calls for c in candidates))
            pool_ids = [candidate.id for candidate in candidates]
            request = {"task_id": task_id, "split": split, "root": str(Path(root).resolve()),
                       "seed": seed, "steps": steps[:prefix], "factory_spec": factory_spec}
            run_id = uuid.uuid4().hex[:16]
            reference = replay_in_fresh_process({**request, "experiment_name": f"{experiment_prefix}_{run_id}_ref"}, timeout)
            expected_observations = [step["output"] for step in steps[:prefix]]
            receipt = {"task_id": task_id, "state_id": state_id, "split": split,
                       "source": source, "prefix_length": prefix,
                       "prefix_sha256": stable_digest(steps[:prefix]),
                       "candidate_count": len(candidates), "accepted_count": 0,
                       "known_label_count": 0, "candidates": []}
            if not reference["ok"] or reference["observations"] != expected_observations:
                receipt["error"] = reference.get("error", "Reference replay observation mismatch")
                receipt["reference_observations_sha256"] = reference.get("observations_sha256")
                rejected += len(candidates)
                state_receipts.append(receipt)
                continue
            receipt["before_fingerprint"] = reference["before_fingerprint"]
            receipt["observations_sha256"] = reference["observations_sha256"]
            for number, candidate in enumerate(candidates):
                result = replay_in_fresh_process({**request, "candidate": candidate.to_dict(),
                    "experiment_name": f"{experiment_prefix}_{run_id}_c{number}"}, timeout)
                candidate_receipt = {"candidate_id": candidate.id, "accepted": False}
                receipt["candidates"].append(candidate_receipt)
                if not result["ok"]:
                    candidate_receipt["error"] = result["error"]
                    rejected += 1
                    continue
                if (result["before_fingerprint"] != reference["before_fingerprint"] or
                    result["observations"] != reference["observations"] or
                    result["before_status"] != reference["before_status"]):
                    candidate_receipt["error"] = "Fork before-state, observation or evaluator mismatch"
                    rejected += 1
                    continue
                before = mapped_goal_status(goals, groups, result["before_status"])
                after = mapped_goal_status(goals, groups, result["after_status"])
                labels = transition_labels(before, after)
                example = Example(task_id=task_id, scenario_id=task_id.split("_")[0],
                    state_id=state_id, split=split, instruction=instruction, history=history,
                    goals=goals, candidate=candidate, labels=labels,
                    source=source, before_fingerprint=reference["before_fingerprint"],
                    metadata={"prefix_length": prefix, "prefix_sha256": receipt["prefix_sha256"],
                              "observations_sha256": reference["observations_sha256"],
                              "mapping_sha256": stable_digest(mapping),
                              "candidate_group_max_cost": pool_max_cost,
                              "candidate_pool_ids": pool_ids,
                              "candidate_pool_size": len(candidates),
                              "after_fingerprint": result["after_fingerprint"],
                              "collection_seed": seed, "fresh_process_replay": True})
                examples.append(example.to_dict())
                candidate_receipt.update({"accepted": True, "known_labels": sum(x != IGNORE_LABEL for x in labels),
                                          "after_fingerprint": result["after_fingerprint"]})
                receipt["accepted_count"] += 1
                receipt["known_label_count"] += candidate_receipt["known_labels"]
            state_receipts.append(receipt)
    receipt = {"schema_version": 1, "evidence_origin": "AppWorld" if factory_spec is None else "synthetic_fixture",
               "status": "no_accepted_examples" if not examples else ("complete_with_rejections" if rejected else "complete"),
               "seed": seed, "num_states": len(state_receipts), "num_examples": len(examples),
               "skipped_no_checkpoints": skipped_no_checkpoints,
               "num_rejected_candidates": rejected,
               "known_label_count": sum(r["known_label_count"] for r in state_receipts),
               "mapping_file_sha256": stable_digest(mappings), "trajectory_files": source_files,
               "dataset_access": "train/dev only", "branching": "one spawned process per full-prefix replay"}
    write_jsonl(output, examples)
    write_jsonl(str(output) + ".states.jsonl", state_receipts)
    atomic_json(str(output) + ".receipt.json", receipt)
    return receipt
