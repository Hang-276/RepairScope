"""Train-only branch execution in one fresh process and world per receipt.

Checker observations inform offline revision only, never online repair prompts. Custom
factories are useful for tests, and always produce synthetic-fixture evidence.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import traceback
from typing import Any
import uuid
from .clock import elapsed_time


ReplayReceipt = dict[str, Any]
IDENTITY_FIELDS = ("environment_sha256", "task_sha256", "goals_sha256",
                   "protocol_sha256", "mapping_sha256")
_HASH = re.compile(r"[0-9a-fA-F]{64}\Z")
_FAILURE = re.compile(r"Traceback \(most recent call last\)|Execution failed|"
                      r"(?:^|\n)\s*(?:[A-Za-z_][A-Za-z_0-9]*Error|"
                      r"[A-Za-z_][A-Za-z_0-9]*Exception):", re.I)


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, bool, float)):
        return value
    raise TypeError(f"Cannot fingerprint {type(value).__name__}")


def stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def semantic_changes(changes: dict[str, Any]) -> dict[str, Any]:
    """Hash ordered SQL and values; discard adapter instrumentation only."""
    result = {}
    for app, entries in sorted(changes.items()):
        normalized = []
        for entry in entries:
            if isinstance(entry, dict):
                if not {"statement", "parameters"} <= entry.keys():
                    raise ValueError("Unknown DB-change envelope")
                entry = [entry["statement"], entry["parameters"], entry.get("executemany", False)]
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                raise ValueError("DB changes require SQL, parameters, executemany")
            normalized.append([str(entry[0]).strip(), _jsonable(entry[1]), bool(entry[2])])
        result[str(app)] = normalized
    return result


def _changes(world: Any) -> dict[str, Any]:
    return semantic_changes(world.models.get_tracked_db_changes())


def _fingerprint(task_id: str, changes: dict[str, Any]) -> str:
    return stable_digest({"task_id": task_id, "changes": changes})


def _resolve(spec: str) -> Any:
    module, name = spec.split(":", 1)
    if not module or not name or "." in name:
        raise ValueError("Use an importable module:function or module:class")
    return getattr(importlib.import_module(module), name)


class AppWorldFactory:
    def __init__(self, root: str | Path, seed: int = 100):
        self.root, self.seed = Path(root).expanduser().resolve(), seed

    @contextmanager
    def create(self, task_id: str, experiment_name: str, split: str,
               full_ground_truth: bool = False):
        if split != "train":
            raise ValueError("Branch learning is train-only")
        os.environ["APPWORLD_ROOT"] = str(self.root)
        from appworld.common.path_store import path_store
        from appworld import AppWorld
        path_store.update_root(str(self.root))
        world = AppWorld(task_id=task_id, experiment_name=experiment_name,
                         random_seed=self.seed, load_ground_truth=True,
                         ground_truth_mode="full", max_interactions=1000)
        try:
            yield world
        finally:
            world.close()


def validate_intervention(intervention: dict[str, Any]) -> None:
    """Conservative syntax gate; semantic legality still needs audited evidence.

    No imports, files, raw DBs, dynamic execution, private attributes, or method
    calls on returned objects are allowed. Queries additionally use read API
    names, and execution must prove the expected observable value.
    """
    if not isinstance(intervention, dict):
        raise ValueError("Intervention must be an object")
    if intervention.get("public_api_only") is not True or intervention.get("semantics_preserved") is not True:
        raise ValueError("Intervention requires public API and semantic-preservation declarations")
    allowed = (ast.Module, ast.Expr, ast.Assign, ast.Name, ast.Load, ast.Store,
               ast.Constant, ast.Call, ast.Attribute, ast.keyword, ast.List,
               ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice, ast.Compare,
               ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In,
               ast.NotIn, ast.Is, ast.IsNot, ast.BoolOp, ast.And, ast.Or,
               ast.UnaryOp, ast.Not, ast.USub, ast.UAdd, ast.BinOp, ast.Add,
               ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.IfExp)
    builtins = {"print", "len", "str", "int", "float", "bool", "list", "tuple",
                "dict", "sorted", "all", "any", "min", "max", "sum"}
    forbidden_names = {"open", "exec", "eval", "compile", "globals", "locals", "vars",
        "getattr", "setattr", "delattr", "world", "models", "db", "database", "sqlite3",
        "connection", "cursor", "os", "sys", "subprocess", "pathlib", "importlib", "builtins"}
    forbidden_attributes = {"models", "db", "database", "connection", "cursor", "execute",
        "executemany", "commit", "rollback", "system", "popen", "open", "unlink", "rmdir"}
    read_prefixes = ("get_", "list_", "search_", "find_", "read_", "fetch_", "count_", "check_", "show_")
    for field in ("code", "query"):
        code = intervention.get(field)
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"Intervention {field} must be nonempty code")
        tree = ast.parse(code)
        api_calls = 0
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                raise ValueError(f"Unsafe intervention syntax: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                raise ValueError("Private intervention names are forbidden")
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                raise ValueError("Raw database/files/system/dynamic execution access is forbidden")
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise ValueError("Private API attributes are forbidden")
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes:
                raise ValueError("Raw database/files/system attributes are forbidden")
            if isinstance(node, ast.Assign) and any(not isinstance(t, ast.Name) or t.id in builtins | {"apis"} for t in node.targets):
                raise ValueError("Only local name assignment is permitted")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in builtins:
                    continue
                parts, current = [], node.func
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if not isinstance(current, ast.Name) or current.id != "apis" or len(parts) != 2:
                    raise ValueError("Only public apis.app.endpoint calls are permitted")
                api_calls += 1
                if field == "query" and not parts[0].startswith(read_prefixes):
                    raise ValueError("Condition query must use a read-only API name")
        if not api_calls and code != "print(None)":
            raise ValueError("Condition intervention/query requires a public API call")
    if not isinstance(intervention.get("expected_output"), str):
        raise ValueError("Intervention expected_output must be text")


def _validate_request(request: dict[str, Any]) -> None:
    if request.get("split") != "train":
        raise ValueError("Branch learning is train-only; dev/test replay is forbidden")
    for field in ("task_id", "scenario_id", "root", "source"):
        if not isinstance(request.get(field), str) or not request[field]:
            raise ValueError(f"Missing {field}")
    identity = request.get("identity", {})
    if not isinstance(identity, dict) or any(not isinstance(identity.get(k), str) or not _HASH.fullmatch(identity[k]) for k in IDENTITY_FIELDS):
        raise ValueError("All five full identity SHA256 hashes are required")
    mapping = request.get("goal_mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Nonempty goal_mapping required")
    if identity["mapping_sha256"].lower() != stable_digest(mapping):
        raise ValueError("mapping_sha256 does not match goal_mapping")
    for goal, group in mapping.items():
        if not isinstance(goal, str) or not isinstance(group, dict) or type(group.get("complete")) is not bool:
            raise ValueError("Invalid goal mapping")
        requirements = group.get("requirements")
        if not isinstance(requirements, list) or any(not isinstance(r, str) or not r for r in requirements) or len(requirements) != len(set(requirements)):
            raise ValueError("Mapping requires unique requirement strings")
    for field in ("g_fix", "g_keep"):
        ids = request.get(field)
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or any(g not in mapping for g in ids):
            raise ValueError(f"Invalid {field}")
    if not request["g_fix"] or set(request["g_fix"]) & set(request["g_keep"]):
        raise ValueError("g_fix must be nonempty and disjoint from g_keep")
    if not isinstance(request.get("steps"), list) or any(not isinstance(s, dict) or not isinstance(s.get("code"), str) or not isinstance(s.get("output"), str) for s in request["steps"]):
        raise ValueError("Every prefix step requires code and exact output")
    if not isinstance(request.get("program"), str):
        raise ValueError("program must be code text")
    compile(request["program"], "<repair>", "exec")
    if request.get("intervention") is not None:
        validate_intervention(request["intervention"])
    if request.get("condition_query") is not None:
        validate_intervention({"code": "print(None)", "query": request["condition_query"],
            "expected_output": "", "public_api_only": True, "semantics_preserved": True})
    if request.get("checker_spec") and not _HASH.fullmatch(str(request.get("checker_audit_sha256", ""))):
        raise ValueError("A custom checker requires a full checker audit SHA256")
    # Membership is checked before any world/import/factory construction. Only
    # the train ID list is read, including for synthetic adapters.
    listing = Path(request["root"]).expanduser().resolve() / "data" / "datasets" / "train.txt"
    ids = [line.split(":", 1)[0].strip() for line in listing.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)) or request["task_id"] not in ids:
        raise ValueError("Task is not uniquely present in official train list")


def _goal_vector(world: Any, request: dict[str, Any], *, intervened: bool = False) -> dict[str, bool | None]:
    mapping = request["goal_mapping"]
    unknown = {g: None for g in mapping}
    if intervened and not (request.get("checker_spec") and request.get("checker_audit_sha256") and request["intervention"].get("checker_invariant") is True):
        return unknown
    original_changes = _changes(world)
    world.save()
    if request.get("checker_spec"):
        values = _resolve(request["checker_spec"])(world, deepcopy(mapping))
        if not isinstance(values, dict) or set(values) != set(mapping) or any(v is not None and type(v) is not bool for v in values.values()):
            raise ValueError("Audited checker must return exactly goal IDs with bool/null values")
    else:
        tracker = world.evaluate()
        data = tracker.to_dict() if hasattr(tracker, "to_dict") else tracker
        if not isinstance(data, dict) or not {"passes", "failures"} <= data.keys():
            raise ValueError("Evaluator did not supply requirement statuses")
        statuses = {}
        for key, truth in (("passes", True), ("failures", False)):
            for entry in data[key]:
                requirement = entry["requirement"]
                if requirement in statuses:
                    raise ValueError("Duplicate evaluator requirement")
                statuses[requirement] = truth
        values = {g: (all(statuses[r] for r in group["requirements"])
                      if all(r in statuses for r in group["requirements"]) else None)
                  for g, group in mapping.items()}
    if _changes(world) != original_changes:
        raise ValueError("Audit checker changed tracked database state")
    owners = {}
    for goal, group in mapping.items():
        for requirement in group["requirements"]:
            owners.setdefault(requirement, []).append(goal)
    shared = {g for goals in owners.values() if len(goals) > 1 for g in goals}
    return {g: (values[g] if group["complete"] and group["requirements"] and g not in shared else None)
            for g, group in mapping.items()}


def _execute(world: Any, code: str) -> tuple[str, str | None]:
    try:
        output = world.execute(code)
        if not isinstance(output, str):
            return "", "Environment execute did not return text"
        return output, "Execution-error observation" if _FAILURE.search(output) else None
    except Exception as error:
        return "", f"{type(error).__name__}: {error}"


def _worker(connection: Any, request: dict[str, Any], receipt: dict[str, Any]) -> None:
    start = elapsed_time()
    try:
        factory = (_resolve(request["factory_spec"])(request["root"], request.get("seed", 100))
                   if request.get("factory_spec") else AppWorldFactory(request["root"], request.get("seed", 100)))
        receipt["process_id"] = os.getpid()
        with factory.create(request["task_id"], receipt["experiment_name"], "train", full_ground_truth=True) as world:
            if getattr(world, "task_id", None) != request["task_id"]:
                raise ValueError("Factory returned a different task")
            observations, fingerprints = [], []
            for index, step in enumerate(request["steps"]):
                # Failed original blocks are intentionally replayed unchanged;
                # their partial writes and Python locals must survive.
                output = world.execute(step["code"])
                if not isinstance(output, str):
                    raise ValueError(f"Prefix observation is not text at step {index}")
                observations.append(output)
                fingerprints.append(_fingerprint(request["task_id"], _changes(world)))
                receipt.update(prefix_observations=list(observations), prefix_fingerprints=list(fingerprints))
                if output != step["output"]:
                    raise ValueError(f"Prefix observation mismatch at step {index}")
            receipt.update(prefix_observations=observations, prefix_fingerprints=fingerprints,
                           observations_sha256=stable_digest(observations), prefix_verified=True)
            initial_changes = _changes(world)
            receipt.update(changes_initial=initial_changes,
                           prefix_fingerprint=_fingerprint(request["task_id"], initial_changes),
                           goals_initial=_goal_vector(world, request))
            intervention = request.get("intervention")
            receipt["intervention_verified"] = intervention is None
            if intervention is not None:
                output, error = _execute(world, intervention["code"])
                receipt.update(intervention_output=output, intervention_error=error,
                               changes_after_intervention=_changes(world))
                if error:
                    raise ValueError("Condition intervention failed: " + error)
                before_query = _changes(world)
                output, error = _execute(world, intervention["query"])
                receipt.update(query_output=output, query_error=error)
                if error or output != intervention["expected_output"]:
                    raise ValueError("Observable condition query did not prove expected output")
                if _changes(world) != before_query:
                    raise ValueError("Condition query changed tracked database state")
                receipt["intervention_verified"] = True
            elif request.get("condition_query") is not None:
                before_query = _changes(world)
                output, error = _execute(world, request["condition_query"])
                receipt.update(query_output=output, query_error=error)
                if error or _changes(world) != before_query:
                    raise ValueError("Baseline condition query failed or changed database state")
            changes_before = _changes(world)
            receipt.update(changes_before=changes_before,
                           before_fingerprint=_fingerprint(request["task_id"], changes_before),
                           goals_before=_goal_vector(world, request, intervened=intervention is not None))
            output, error = _execute(world, request["program"])
            receipt.update(program_output=output, program_error=error,
                           changes_after=_changes(world),
                           goals_after=_goal_vector(world, request, intervened=intervention is not None))
            receipt["after_fingerprint"] = _fingerprint(request["task_id"], receipt["changes_after"])
            receipt["checker_valid"] = all(v is not None for v in receipt["goals_before"].values()) and all(v is not None for v in receipt["goals_after"].values())
            receipt["ok"] = error is None
            if error:
                receipt["errors"].append(error)
    except BaseException as error:
        receipt["ok"] = False
        receipt["errors"].append(f"{type(error).__name__}: {error}")
        receipt["traceback"] = traceback.format_exc()
    finally:
        receipt["duration_seconds"] = elapsed_time() - start
        try:
            connection.send(receipt)
        finally:
            connection.close()


def run_branch(request: dict[str, Any], timeout: float = 300) -> ReplayReceipt:
    """Replay a fixed prefix, optional intervention, and repair in a fresh spawn.

    Failures return receipts with ok=false; no branch reuses or writes a caller's
    live world. Timeouts are terminated and cannot establish a repair witness.
    """
    start = elapsed_time()
    request = json.loads(json.dumps(request))
    origin = "synthetic_fixture" if request.get("factory_spec") or request.get("source") == "synthetic_fixture" else "AppWorld"
    receipt = {"schema_version": 1, "ok": False, "errors": [],
        "task_id": request.get("task_id"), "scenario_id": request.get("scenario_id"),
        "split": request.get("split"), "seed": request.get("seed", 100),
        "identity": request.get("identity", {}), "evidence_origin": origin,
        "source": request.get("source"),
        "branch_source": "cf_intervention" if request.get("intervention") is not None else request.get("source"),
        "original_source": request.get("source"), "g_fix": request.get("g_fix", []),
        "g_keep": request.get("g_keep", []), "goal_mapping": request.get("goal_mapping", {}),
        "program": request.get("program"), "steps_sha256": stable_digest(request.get("steps")),
        "program_sha256": stable_digest(request.get("program")),
        "intervention": request.get("intervention"),
        "checker_audit_sha256": request.get("checker_audit_sha256"),
        "checker_spec": request.get("checker_spec"),
        "event_id": request.get("event_id"), "state_id": request.get("state_id"),
        "cache_key": stable_digest(request), "experiment_name": "repairscope_branch_" + uuid.uuid4().hex,
        "goals_initial": {}, "goals_before": {}, "goals_after": {}, "checker_valid": False,
        "prefix_verified": False}
    try:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        _validate_request(request)
    except Exception as error:
        receipt["errors"].append(f"{type(error).__name__}: {error}")
        receipt["duration_seconds"] = elapsed_time() - start
        return receipt
    context = mp.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(writer, request, receipt))
    try:
        process.start()
    except Exception as error:
        writer.close()
        reader.close()
        receipt["errors"].append(f"Could not spawn replay process: {type(error).__name__}: {error}")
        receipt["wall_seconds"] = elapsed_time() - start
        return receipt
    writer.close()
    try:
        if reader.poll(timeout):
            try:
                receipt = reader.recv()
            except EOFError:
                receipt["errors"].append("Replay process exited without a receipt")
        else:
            receipt["errors"].append(f"Replay timed out after {timeout}s")
        process.join(1)
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            receipt["ok"] = False
            receipt["errors"].append("Replay process did not exit cleanly")
        elif process.exitcode != 0:
            receipt["ok"] = False
            receipt["errors"].append(f"Replay process exit code {process.exitcode}")
    finally:
        reader.close()
    receipt["wall_seconds"] = elapsed_time() - start
    return receipt


def receipt_success(receipt: dict[str, Any]) -> bool:
    """Exact complete recovery and preservation, with no execution/checker gaps."""
    if not receipt.get("ok") or receipt.get("errors") or receipt.get("program_error") or not receipt.get("checker_valid") or not receipt.get("intervention_verified") or not receipt.get("prefix_verified"):
        return False
    mapping = receipt.get("goal_mapping", {})
    initial, before, after = (receipt.get(k, {}) for k in ("goals_initial", "goals_before", "goals_after"))
    fix, keep = receipt.get("g_fix", []), receipt.get("g_keep", [])
    if not fix or set(fix) & set(keep):
        return False
    if any(not mapping.get(g, {}).get("complete") or not mapping.get(g, {}).get("requirements") for g in fix + keep):
        return False
    return (all(after.get(g) is True and before.get(g) is not None for g in fix)
            and all(initial.get(g) is True and before.get(g) is True and after.get(g) is True for g in keep))
