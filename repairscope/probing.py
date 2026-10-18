"""Costed train-only counterfactual probing with prequential replay updates.

Candidate plans contain legal prefix/condition operations, never edited state
tokens. A four-cell plan has two ``conditions``; a simple plan has none. Each
condition is a backend-validated {actions,query,expected_output} specification.
Continuation calls replay complete prefixes in fresh processes and are all billed.
"""
from __future__ import annotations

import copy
import json
import math
import random
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path

from .actor import create_actor, normalize_action, public_goals, public_history, public_state
from .control import BudgetExceeded, BudgetLedger, RepairController, evaluate_goals, score_prediction
from .schema import effective_protected


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _variance(values):
    mean = _mean(values)
    return _mean([(value - mean) ** 2 for value in values])


def _bound(value):
    return min(1.0, max(0.0, float(value)))


def _typed_hypothesis_disagreement(hypotheses):
    """MC effect disagreement survives a deterministic pre-attention scope head."""
    if len(hypotheses) < 2:
        return 0.0
    differences = []
    keys = set().union(*(h.get("change_probabilities", {}).keys() for h in hypotheses))
    for key in keys:
        differences.append(4 * _variance([float(h.get("change_probabilities", {}).get(key, 0)) for h in hypotheses]))
    for left_index, left in enumerate(hypotheses):
        for right in hypotheses[left_index + 1:]:
            a = {f["key"]: f for f in left.get("next_state", {}).get("fields", [])}
            b = {f["key"]: f for f in right.get("next_state", {}).get("fields", [])}
            for key in set(a) | set(b):
                x, y = a.get(key), b.get(key)
                if x is None or y is None or (x.get("known"), x.get("exists")) != (y.get("known"), y.get("exists")):
                    differences.append(1.0)
                elif x.get("known", True) and x.get("exists", True):
                    if x.get("kind") == "number" and x.get("value") is not None and y.get("value") is not None:
                        scale = max(float(x.get("scale", 1)), float(y.get("scale", 1)), 1e-6)
                        differences.append(min(1., abs(float(x["value"]) - float(y["value"])) / scale))
                    else:
                        differences.append(float(type(x.get("value")) is not type(y.get("value")) or x.get("value") != y.get("value")))
            # Even a shared argmax can have different typed-operation confidence.
            p = {item["key"]: float(item.get("probability", 1)) for item in left.get("patch", [])}
            q = {item["key"]: float(item.get("probability", 1)) for item in right.get("patch", [])}
            differences.extend(abs(p.get(key, 0) - q.get(key, 0)) for key in set(p) | set(q))
    return _bound(_mean(differences))


def plan_cost(plan):
    """Conservative explicit bound; backend receipts replace estimates in logs."""
    if "estimated_cost" in plan:
        return max(float(plan["estimated_cost"]), 1e-6)
    prefix = [e.get("action", e) for e in plan.get("prefix", [])]
    continuation = plan.get("continuation", [])
    total = 0.0
    for condition in plan.get("conditions") or [None]:
        setup = list((condition or {}).get("actions", []))
        if (condition or {}).get("query"):
            setup.append(condition["query"])
        for target in (plan["action"], plan["reference"]):
            sequence = prefix + setup + [target]
            total += sum(float(a.get("cost", 1)) for a in sequence)
            for following in continuation:
                sequence.append(following)
                total += sum(float(a.get("cost", 1)) for a in sequence)
    return max(total, 1e-6)


def _plan_step(plan, condition=None):
    """Use one logical prefix clock for planning, prediction, and branch labels.

    Explicit condition/plan steps override inferred prefixes. Condition setup
    receipts do not silently advance this clock: offline collection likewise
    labels its target by the original prefix length.
    """
    condition = condition or {}
    step = condition.get("step", plan.get("step"))
    if step is None:
        history = condition.get("history", plan.get("history", []))
        step = max(len(plan.get("prefix", [])), len(history))
    if type(step) is not int or step < 0:
        raise ValueError("Counterfactual plan step must be a nonnegative integer")
    return step


def _decision_hypothesis_disagreement(inputs, predictions, goals, step, config):
    """Rank the same action/reference candidates under same-checkpoint MC draws."""
    winner_disagreements, ranking_uncertainties = [], []
    for offset in range(0, len(inputs), 2):
        pair = predictions[offset:offset + 2]
        if len(pair) != 2:
            continue
        count = min(len(prediction.get("mc_predictions", [])) for prediction in pair)
        if count < 2:
            continue
        margins, winners = [], []
        for draw in range(count):
            scores = [float(score_prediction(inputs[offset + index][0],
                prediction["mc_predictions"][draw], goals, inputs[offset + index][1],
                step[offset + index] if isinstance(step, (list, tuple)) else step,
                config)["score"]) for index, prediction in enumerate(pair)]
            margins.append(scores[0] - scores[1])
            winners.append(int(scores[1] > scores[0]))
        frequency = max(winners.count(0), winners.count(1)) / count
        winner_disagreements.append(2 * (1 - frequency))
        spread = math.sqrt(_variance(margins))
        ranking_uncertainties.append(spread / (abs(_mean(margins)) + spread + 1e-6))
    return _bound(_mean(winner_disagreements)), _bound(_mean(ranking_uncertainties))


def candidate_statistics(plan, predict, damage_scale=1.0, config=None):
    config = config or {}
    goals = public_goals(plan.get("goals", []))
    conditions = plan.get("conditions") or [{"state": plan["state"]}]
    states = [public_state(condition["state"]) for condition in conditions]
    inputs = [(state, action) for state in states for action in (plan["action"], plan["reference"])]
    predictions, input_steps = [], []
    for index, (state, action) in enumerate(inputs):
        condition = conditions[index // 2]
        request = {"state": state, "action": normalize_action(action),
                   "history": public_history(condition.get("history", plan.get("history", []))), "goals": goals}
        request["step"] = _plan_step(plan, condition)
        input_steps.append(request["step"])
        predictions.append(predict(request))
    goal_sufficient = bool(config.get("model", {}).get("goal_sufficient", False) or
                           any("goal_escape" in prediction for prediction in predictions))
    def risk_probability(prediction):
        return prediction.get("goal_escape", prediction.get("escape", 0)) if goal_sufficient else prediction.get("escape", 0)
    uncertainty = _mean([max(_bound(p.get("uncertainty", 0)),
                            _typed_hypothesis_disagreement(p.get("mc_predictions", [])),
                            4 * _variance([float(risk_probability(h)) for h in p.get("mc_predictions", [])]))
                         for p in predictions])
    risk = max(_bound(risk_probability(p)) for p in predictions)
    decision_disagreement, ranking_uncertainty = (0.0, 0.0)
    if goal_sufficient:
        decision_disagreement, ranking_uncertainty = _decision_hypothesis_disagreement(
            inputs, predictions, goals, input_steps, config.get("controller", {}))
    # Competing hypotheses are dropout forwards of this very same checkpoint.
    disagreement = []
    for prediction in predictions:
        typed_disagreement = _typed_hypothesis_disagreement(prediction.get("mc_predictions", []))
        hypotheses = prediction.get("mc_scope_probabilities", [])
        if hypotheses:
            keys = set().union(*(h.keys() for h in hypotheses))
            scope_disagreement = _mean([4 * _variance([float(h.get(key, 0)) for h in hypotheses]) for key in keys])
            disagreement.append(max(typed_disagreement, scope_disagreement))
        else:
            # MC uncertainty is still a shared-model disagreement statistic.
            disagreement.append(max(typed_disagreement, _bound(prediction.get("uncertainty", 0))))
    damage = 0.0
    weight_sum = 0.0
    for i, goal in enumerate(goals):
        active = [effective_protected(goal, step, True) if goal_sufficient else bool(goal.get("protected"))
                  for step in input_steps]
        if any(active):
            weight = float(goal.get("weight", 1))
            weight_sum += weight
            for (state, _), prediction, enabled in zip(inputs, predictions, active):
                if not enabled or evaluate_goals(state, goals)[i] is not True:
                    continue
                after = evaluate_goals(prediction.get("next_state", {}), goals, previous=state)
                probs = prediction.get("goal_probabilities", [])
                possibility = (float(probs[i][2]) if i < len(probs) else float(after[i] is False))
                # An omitted protected field remains at risk even if the hard patch is unchanged.
                scope_prob = float(prediction.get("scope_probabilities", {}).get(goal.get("field"), 0))
                damage += weight * max(possibility, risk * (1 - scope_prob)) / len(predictions)
    result = {"U": _bound(0.5 * uncertainty + 0.5 * risk), "D": _bound(_mean(disagreement)),
              "R": _bound(damage / max(weight_sum * float(damage_scale), 1e-6)),
              "C": plan_cost(plan), "predictions": predictions}
    if goal_sufficient:
        result.update({"D": max(result["D"], decision_disagreement, ranking_uncertainty),
                       "R": max(result["R"], risk * max(decision_disagreement, ranking_uncertainty)),
                       "goal_decision_risk": risk, "decision_disagreement": decision_disagreement,
                       "ranking_uncertainty": ranking_uncertainty,
                       "risk_semantics": "goal_escape_when_available_else_state_escape"})
    return result


def selection_probabilities(statistics, strategy="risk", epsilon=0.1, novelty_counts=None):
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must lie in [0,1]")
    if not statistics:
        raise ValueError("No candidate experiments")
    novelty_counts = novelty_counts or [0] * len(statistics)
    scores = []
    for i, stat in enumerate(statistics):
        if strategy == "random":
            score = 1.0
        elif strategy == "uncertainty":
            score = stat["U"] / max(stat["C"], 1e-6)
        elif strategy in {"novelty", "novelty_learnability", "onlinewm"}:
            # Favor new, moderately uncertain records (the common learnability heuristic).
            u = stat["U"]
            score = (1 / math.sqrt(1 + novelty_counts[i])) * (4 * u * (1 - u)) / max(stat["C"], 1e-6)
        elif strategy in {"risk", "repairscope"}:
            score = stat["U"] * stat["D"] * stat["R"] / max(stat["C"], 1e-6)
        else:
            raise ValueError(f"Unknown probing strategy {strategy}")
        scores.append(max(0.0, score))
    denominator, n = sum(scores), len(scores)
    exploitation = [s / denominator for s in scores] if denominator else [1 / n] * n
    return [(1 - epsilon) * p + epsilon / n for p in exploitation]


class ExperienceReplay:
    def __init__(self, capacity=10000, seed=0):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity, self.rng = int(capacity), random.Random(seed)
        self.buckets = {"old": [], "new": [], "escape": []}

    def add(self, records, old=False):
        for record in records:
            if record.get("split") != "train" or record.get("frozen_query", False):
                raise ValueError("Only training records may enter replay; frozen queries never return")
            sample = copy.deepcopy(record)
            self.buckets["old" if old else "new"].append(sample)
            audit = sample.get("audit", {})
            changes = audit.get("changes", [])
            keys = {c if isinstance(c, str) else c.get("key", c.get("field")) for c in changes}
            if keys - set(sample.get("scope", [])):
                self.buckets["escape"].append(sample)
            # Keep each stratum bounded without silently evicting all old evidence.
            quota = max(1, self.capacity // 3)
            for bucket in self.buckets.values():
                if len(bucket) > quota:
                    del bucket[:len(bucket) - quota]

    def sample(self, count):
        available = [bucket for bucket in self.buckets.values() if bucket]
        if not available:
            return []
        unique = {}
        for bucket in available:
            for record in bucket:
                signature = (record.get("trajectory_id"), record.get("step"), record.get("cf_pair_id"),
                             record.get("cf_condition"), record.get("cf_role"))
                unique[signature] = record
        units = defaultdict(list)
        for signature, record in unique.items():
            unit = ("pair", record["cf_pair_id"]) if record.get("cf_pair_id") else ("natural", signature)
            units[unit].append(record)
        # Draw whole sibling units, retaining boundary/effect supervision in replay.
        strata = []
        for bucket in available:
            keys = {("pair", r["cf_pair_id"]) if r.get("cf_pair_id") else ("natural", (
                    r.get("trajectory_id"), r.get("step"), r.get("cf_pair_id"), r.get("cf_condition"), r.get("cf_role")))
                    for r in bucket}
            ordered_stratum = sorted(keys, key=repr)
            self.rng.shuffle(ordered_stratum)
            strata.append(ordered_stratum)
        ordered, seen = [], set()
        while any(strata):
            for stratum in strata:
                if stratum:
                    unit = stratum.pop()
                    if unit not in seen:
                        seen.add(unit)
                        ordered.append(unit)
        result = []
        for unit in ordered:
            result.extend(copy.deepcopy(units[unit]))
            if len(result) >= int(count):
                break
        return result


def _load_plans(config):
    plans = config.get("probing", {}).get("plans")
    if isinstance(plans, str):
        payload = Path(plans).read_text()
        try:
            plans = json.loads(payload)
        except json.JSONDecodeError:
            plans = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if isinstance(plans, dict):
        plans = plans.get("plans", [])
    if plans:
        normalized = copy.deepcopy(plans)
        for index, plan in enumerate(normalized):
            if not plan.get("id"):
                plan["id"] = str(plan.get("pair_id", f"{plan.get('task_id', 'task')}-{index}"))
            if isinstance(plan.get("conditions"), dict):
                if not set(plan["conditions"]) <= {"c0", "c1"}:
                    raise ValueError("Condition dictionary keys must be c0 and/or c1")
                plan["conditions"] = [plan["conditions"][name] for name in ("c0", "c1") if name in plan["conditions"]]
        return normalized
    raise ValueError("Probing requires probing.plans with legal action/reference, observed state, task_id, prefix and optional conditions; synthetic token interventions are forbidden")


def _branch_records(env, plan, predictor, ledger, selection_id):
    records, receipts = [], []
    conditions = plan.get("conditions") or [None]
    if len(conditions) not in {1, 2}:
        raise ValueError("Experiments must contain exactly 2 or 4 branches")
    for c_index, condition in enumerate(conditions):
        for role in ("action", "reference"):
            prefix = copy.deepcopy(plan.get("prefix", []))
            branch_history = copy.deepcopy((condition or {}).get("history", plan.get("history", [])))
            action = normalize_action(plan[role])
            base_step = _plan_step(plan, condition)
            for continuation_index, target in enumerate([action] + plan.get("continuation", [])):
                call_condition = condition if continuation_index == 0 else None
                # The public prediction is committed before receiving any branch labels.
                before = public_state((condition or {}).get("state", plan["state"])) if continuation_index == 0 else public_state(result["state"])
                prediction = predictor({"state": before, "action": target,
                    "history": public_history(branch_history), "goals": public_goals(plan.get("goals", [])),
                    "step": base_step + continuation_index})
                prediction_time = time.monotonic()
                expected = sum(float(e.get("action", e).get("cost", 1)) for e in prefix) + float(target.get("cost", 1))
                if call_condition:
                    expected += sum(float(a.get("cost", 1)) for a in call_condition.get("actions", []))
                    if call_condition.get("query"):
                        expected += float(call_condition["query"].get("cost", 1))
                event = ledger.charge("branch", expected, role=role, condition=c_index, continuation=continuation_index)
                try:
                    result = env.branch(prefix, target, condition=call_condition, max_cost=ledger.remaining + expected)
                except Exception as error:
                    charged = re.search(r"charged_cost[=:]\s*([0-9.]+)", str(error))
                    if charged:
                        ledger.settle("failed_branch_actual_adjustment", float(charged.group(1)), expected)
                    event.update(valid=False, error=type(error).__name__, cost_uncertain=charged is None)
                    # Unknown partial execution cost is not safe to silently estimate and continue.
                    raise RuntimeError("Branch failed without a complete cost receipt; probing stopped") from error
                actual = float(result.get("cost", expected))
                event.update(actual_cost=actual, valid=result.get("valid", True), cost_breakdown=result.get("cost_breakdown", {}))
                delta = actual - expected
                if delta > ledger.remaining:
                    # Preserve every incurred call even when a backend exceeds its supplied bound.
                    ledger.spent += delta
                    ledger.events.append({"kind": "branch_actual_overrun", "units": delta, "cumulative_cost": ledger.spent})
                    raise RuntimeError("Actual branch cost exceeded reserved budget; all incurred costs logged and probing stopped")
                if delta > 0:
                    ledger.charge("branch_actual_extra", delta)
                elif delta < 0:
                    ledger.spent += delta
                    event["units"] = actual
                    event["cumulative_cost"] = ledger.spent
                truth_before = result.get("before_state", before)
                # Condition-state features require legal execution. Store the earlier prediction
                # separately if the plan did not supply that already-observed condition state.
                prediction_matches = public_state(truth_before) == before
                state = public_state(truth_before)
                goals = public_goals(plan.get("goals", getattr(env, "metadata", {}).get("goals", [])))
                record = {"task_id": str(plan["task_id"]), "source_task_id": str(plan["task_id"]),
                    "group_id": str(plan.get("group_id", plan["task_id"])), "split": "train", "source_split": "train",
                    "evidence_origin": result.get("evidence_origin", getattr(env, "evidence_origin", "unknown")),
                    "step": base_step + continuation_index, "trajectory_id": f"{selection_id}-{c_index}-{role}",
                    "state": state, "action": target, "next_state": result["state"],
                    "history": public_history(branch_history), "goals": goals,
                    "goal_before": evaluate_goals(state, goals), "goal_after": evaluate_goals(result["state"], goals, previous=state),
                    "scope": prediction.get("scope", []) if prediction_matches else [],
                    "prediction": prediction, "prequential_prediction_valid": prediction_matches,
                    "predicted_at_monotonic": prediction_time,
                    "scope_model_version": selection_id, "action_valid": result.get("valid", True),
                    "audit": {**result.get("audit", {}), "model_version": selection_id},
                    "audit_state": result.get("audit_before_state", truth_before),
                    "audit_next_state": result.get("audit_state", result.get("state")), "used_for_training": True,
                    "cf_pair_id": f"{selection_id}-step-{continuation_index}",
                    "cf_condition": f"c{c_index}", "cf_role": role, "actual_cost": actual}
                if not prediction_matches:
                    raise ValueError("Legally replayed branch state differs from its logged prediction input; provide previously observed condition.state and exact prefix")
                record["goal_before"] = evaluate_goals(record["audit_state"], goals)
                record["goal_after"] = evaluate_goals(record["audit_next_state"], goals, previous=record["audit_state"])
                records.append(record)
                receipts.append(event)
                if continuation_index == 0 and condition:
                    condition_receipts = result.get("condition_receipts")
                    if condition_receipts:
                        prefix.extend(copy.deepcopy(condition_receipts))
                    elif record["evidence_origin"] == "appworld":
                        raise ValueError("Real continuation replay requires condition action/output/fingerprint receipts")
                    else:
                        prefix.extend({"action": setup} for setup in condition.get("actions", []))
                        if condition.get("query"):
                            prefix.append({"action": condition["query"], "output": condition["expected_output"]})
                prefix.append({"action": target, "output": result.get("observation", ""), "fingerprint": result.get("fingerprint")})
                branch_history.append({"state": state, "action": target})
    return records, receipts


def _discovered_associations(records):
    """Separate audited scope escapes from conditional action/reference effects."""
    associations, pairs = set(), defaultdict(list)
    for record in records:
        action = record["action"]["name"]
        for key in set(record.get("audit", {}).get("changes", [])) - set(record.get("scope", [])):
            associations.add(("scope_escape", action, str(key), record.get("cf_condition", "natural")))
        if record.get("cf_pair_id"):
            pairs[record["cf_pair_id"]].append(record)
    from .schema import cf_four_grid_targets
    for siblings in pairs.values():
        if len(siblings) != 4:
            continue
        targets = cf_four_grid_targets(siblings)
        action = next(r["action"]["name"] for r in siblings if r["cf_role"] == "action")
        for key, y0, y1, valid in zip(targets["keys"], targets["y0"], targets["y1"], targets["mask"]):
            if valid and y0 != y1:
                associations.add(("conditional_boundary", action, key, f"{int(y0)}->{int(y1)}"))
    return associations


def _prepare_plans(config, plans, ledger, experiment=None):
    """Obtain missing condition observations through real legal read branches."""
    from .environment import create_environment
    prepared = []
    experiment = experiment or "probe-prepare-" + uuid.uuid4().hex[:12]
    for index, plan in enumerate(plans):
        env = create_environment(config, str(plan["task_id"]), "train", f"{experiment}-prepare-{index}")
        try:
            initial_state = env.reset()
            plan.setdefault("goals", public_goals(getattr(env, "metadata", {}).get("goals", [])))
            for condition in plan.get("conditions") or [None]:
                target = condition if condition is not None else plan
                if "state" in target:
                    continue
                if condition is None and not plan.get("prefix"):
                    target["state"] = public_state(initial_state)
                    target["state_provenance"] = {"legally_observed": True, "source": "initial_reset"}
                    continue
                checks = getattr(env, "metadata", {}).get("read_only_actions", [])
                query = (condition or {}).get("query") or (checks[0] if checks else None)
                if not query:
                    raise ValueError("Missing legal prefix state: provide a public read action or an already observed state")
                if ledger.remaining < 1:
                    raise BudgetExceeded("Candidate preparation exhausted the execution budget")
                try:
                    result = env.branch(plan.get("prefix", []), query, condition=condition, max_cost=ledger.remaining)
                except Exception as error:
                    match = re.search(r"charged_cost[=:]\s*([0-9.]+)", str(error))
                    if match:
                        ledger.charge("prepare_failed", float(match.group(1)))
                    raise RuntimeError("Legal candidate preparation failed; no state tokens were invented") from error
                ledger.charge("candidate_observation", float(result["cost"]), plan_id=plan.get("pair_id", index),
                              cost_breakdown=result.get("cost_breakdown", {}))
                if not result.get("valid", True):
                    raise ValueError("Candidate condition observation action was invalid")
                from .schema import semantic_diff
                if semantic_diff(result.get("before_state", result["state"]), result["state"]) or result.get("audit", {}).get("changes"):
                    raise ValueError("Candidate preparation query changed observed state")
                target["state"] = public_state(result.get("before_state", result["state"]))
                target["state_provenance"] = {"legally_executed": True, "prefix_verified": result.get("prefix_verified", True),
                                               "evidence_origin": result.get("evidence_origin", getattr(env, "evidence_origin", "unknown"))}
            plan.setdefault("state", copy.deepcopy(plan["conditions"][0]["state"]) if plan.get("conditions") else None)
            prepared.append(plan)
        finally:
            env.close()
    return prepared


def _frozen_queries(config, task_ids, predictor, output_dir):
    from .environment import create_environment
    from .metrics import aggregate_runs
    from .evaluation import create_run_context
    run_context = create_run_context(output_dir, config, "probe-query")
    query_config = copy.deepcopy(config)
    query_config.setdefault("environment", {})["offline_audit"] = False
    split = config.get("probing", {}).get("query_split", "dev")
    if split == "train":
        raise ValueError("Frozen query evaluation must use a held-out split")
    actor, results = create_actor(config), []
    for task_id in task_ids:
        env = create_environment(query_config, task_id, split, run_context["experiment"])
        try:
            controller_options = {**config.get("controller", {}),
                                  "use_goal_head": float(config.get("loss", {}).get("goal", 1)) > 0}
            result = RepairController(actor, predictor, controller_options).run(env)
            result.update(task_id=task_id, group_id=str(task_id).split("_")[0], official=env.evaluate(),
                          frozen_query=True, used_for_training=False, evidence_origin=getattr(env, "evidence_origin", "unknown"))
            results.append(result)
        finally:
            env.close()
    summary = aggregate_runs(results)
    summary.update(run_context)
    environment_kinds = {"environment", "environment_actual_adjustment", "failed_environment_actual_adjustment"}
    summary["environment_execution_cost"] = sum(
        float(event["units"]) for result in results for event in result["budget"]["events"]
        if event["kind"] in environment_kinds)
    summary["actor_calls"] = sum(event["kind"] == "actor" for result in results for event in result["budget"]["events"])
    summary["predictor_calls"] = sum(event["kind"] == "prediction" for result in results for event in result["budget"]["events"])
    summary["checker_evaluations"] = len(results)
    summary["official_checker_evaluations"] = sum(result.get("official", {}).get("evidence_origin") == "appworld" for result in results)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (Path(output_dir) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_probing(config, checkpoint, output_dir):
    from .environment import create_environment, official_task_ids
    from .training import load_predictor, update_predictor, save_predictor
    from .model import predict_record
    from .data import read_jsonl, write_jsonl
    from .calibration import apply_calibration
    config = apply_calibration(config, checkpoint)
    config.setdefault("environment", {})["offline_audit"] = True
    options = config.get("probing", {})
    disabled = float(options.get("budget_calls", 2000)) == 0
    plans = [] if disabled else _load_plans(config)
    query_tasks = set(map(str, options.get("query_task_ids", [])))
    if not query_tasks and config.get("environment", {}).get("backend", "appworld") == "appworld":
        root = config.get("environment", {}).get("root")
        if not root:
            raise ValueError("AppWorld probing needs environment.root to resolve the frozen official dev query split")
        query_tasks = set(official_task_ids(root, options.get("query_split", "dev")))
    query_groups = {task.split("_")[0] for task in query_tasks}
    if query_tasks & {str(plan["task_id"]) for plan in plans}:
        raise ValueError("Frozen query tasks overlap probing source tasks")
    if query_groups & {str(plan.get("group_id", str(plan["task_id"]).split("_")[0])).split("_")[0] for plan in plans}:
        raise ValueError("Frozen query task families overlap probing source groups")
    for plan in plans:
        if plan.get("split", "train") != "train":
            raise ValueError("Counterfactual exploration is train-only")
        if config.get("environment", {}).get("backend", "appworld") == "appworld" and any(
                "output" not in item or not item.get("fingerprint") for item in plan.get("prefix", [])):
            raise ValueError("Real experiment prefixes require saved action/output/fingerprint receipts")
        if len(plan.get("conditions", [])) not in {0, 1, 2}:
            raise ValueError("A plan must use two branches or a legal four-cell condition grid")
        normalize_action(plan["action"])
        normalize_action(plan["reference"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    from .evaluation import create_run_context
    run_context = create_run_context(output, config, "probe")
    device = config.get("train", {}).get("device", config.get("device", "cpu"))
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(options.get("seed", config.get("train", {}).get("seed", config.get("seed", 0))))
    random.seed(seed)
    torch.manual_seed(seed)
    model, tensorizer, model_config = load_predictor(checkpoint, device)
    manifest = model_config.get("checkpoint_manifest", {})
    origins = manifest.get("evidence_origin", [])
    origins = [origins] if isinstance(origins, str) else origins
    if query_groups & {str(group).split("_")[0] for group in manifest.get("groups", [])}:
        raise ValueError("Frozen query task families occur in the starting checkpoint's training manifest")
    if config.get("environment", {}).get("backend", "appworld") == "appworld" and (
            manifest.get("backend") == "synthetic" or any("synthetic" in str(o) or "fixture" in str(o) for o in origins)):
        raise ValueError("Synthetic/fixture checkpoints cannot initialize AppWorld probing experiments")
    online_config = {**config, "model": copy.deepcopy(model_config["model"]), "checkpoint_manifest": manifest}
    if hasattr(model, "use_escape"):
        model.use_escape = float(online_config.get("loss", {}).get("escape", 1)) > 0
    replay = ExperienceReplay(int(options.get("replay_capacity", 10000)), seed)
    old_path = config.get("data", {}).get("train")
    old = []
    if old_path and not disabled:
        old = read_jsonl(old_path)
        if query_tasks & {str(r["task_id"]) for r in old}:
            raise ValueError("Frozen query data is present in training replay")
        if query_groups & {str(r.get("group_id", str(r["task_id"]).split("_")[0])).split("_")[0] for r in old}:
            raise ValueError("Frozen query task families occur in old training replay")
        from .data import validate_dataset
        validate_dataset(old, allow_synthetic=bool(config.get("data", {}).get("allow_synthetic", False)))
        replay.add(old, old=True)
    ledger = BudgetLedger(float(options.get("budget_calls", 2000)))
    rng, novelty = random.Random(seed), defaultdict(int)
    selections, records, updates, optimizer = [], [], [], None
    mc = int(options.get("mc_samples", 4))
    model_calls = {"predictions": 0, "forwards": 0, "latency_seconds": 0.0}
    def predict(record):
        start = time.monotonic()
        model_calls["predictions"] += 1
        model_calls["forwards"] += 1 + (mc if mc > 1 else 0)
        try:
            return predict_record(model, tensorizer, record, device=device, mc_samples=mc)
        finally:
            model_calls["latency_seconds"] += time.monotonic() - start
    thresholds = sorted(set(float(f) * ledger.limit for f in options.get("checkpoints", [.25, .5, .75, 1])))
    curves = []
    known_associations = _discovered_associations(old)
    discovered_at = {association: -1 for association in known_associations}
    new_associations, confirmations = set(), set()
    iteration, stop, query_cost, query_execution_cost, latest_checkpoint = 0, "disabled" if disabled else "budget", 0.0, 0.0, str(checkpoint)
    last_query, last_query_update = None, None
    try:
        if query_tasks:
            last_query = _frozen_queries(config, sorted(query_tasks), predict, output / "initial-queries")
            last_query_update = 0
            query_cost += float(last_query.get("cost", 0))
            query_execution_cost += float(last_query.get("environment_execution_cost", 0))
            curves.append({"budget_threshold": 0.0, "actual_cost": 0.0, "updates": 0,
                           "checkpoint": str(checkpoint), "frozen_query": last_query, "initial": True})
        if ledger.remaining:
            plans = _prepare_plans(config, plans, ledger, experiment=run_context["experiment"])
            (output / "prepared_plans.json").write_text(json.dumps(plans, ensure_ascii=False, indent=2))
        while True:
            affordable = [plan for plan in plans if "state" in plan and plan_cost(plan) <= ledger.remaining]
            if not affordable:
                if not disabled:
                    stop = "budget" if ledger.remaining <= 1e-9 else "pool_unaffordable"
                break
            statistics = [candidate_statistics(plan, predict, options.get("damage_scale", 1), online_config) for plan in affordable]
            probabilities = selection_probabilities(statistics, options.get("strategy", "risk"), float(options.get("epsilon", .1)),
                [novelty[str(plan.get("id", plan["task_id"]))] for plan in affordable])
            index = rng.choices(range(len(affordable)), probabilities)[0]
            plan, statistic = affordable[index], statistics[index]
            identity = str(plan.get("id", plan["task_id"]))
            novelty[identity] += 1
            execution_experiment = f"{run_context['experiment']}-iteration-{iteration}"
            env = create_environment(config, str(plan["task_id"]), "train", execution_experiment)
            try:
                new, receipts = _branch_records(env, plan, predict, ledger, f"probe-{iteration}")
            finally:
                env.close()
            from .collection import _align_counterfactual_audits
            _align_counterfactual_audits(new)
            # Log predictions before any update; complete sibling groups enter replay together.
            selection = {"iteration": iteration, "plan_id": identity, "selected_probability": probabilities[index],
                         "candidate_probabilities": probabilities, "candidate_ids": [str(p.get("id", p["task_id"])) for p in affordable],
                         "statistics": {k: v for k, v in statistic.items() if k != "predictions"},
                         "actual_cost": sum(r["actual_cost"] for r in receipts), "cumulative_cost": ledger.spent,
                         "branch_receipts": receipts, "predictor_updated_after": False}
            selection["experiment"] = execution_experiment
            discovered = _discovered_associations(new)
            novel = discovered - known_associations
            confirmations.update(association for association in discovered if association in new_associations and discovered_at[association] < iteration)
            for association in novel:
                discovered_at[association] = iteration
            known_associations.update(discovered)
            new_associations.update(novel)
            selection.update(discoveries=[list(a) for a in sorted(discovered)],
                             new_discoveries=[list(a) for a in sorted(novel)])
            selections.append(selection)
            records.extend(new)
            write_jsonl(output / "transitions.jsonl", records)
            (output / "selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2))
            replay.add(new)
            iteration += 1
            if iteration % int(options.get("update_every", 16)) == 0:
                batch = replay.sample(int(options.get("replay_batch_size", 64)))
                update = update_predictor(model, tensorizer, batch, online_config, optimizer=optimizer)
                optimizer = update.pop("optimizer", optimizer)
                updates.append({"iteration": iteration, **update})
                selection["predictor_updated_after"] = True
            while thresholds and ledger.spent >= thresholds[0]:
                threshold = thresholds.pop(0)
                curve_path = output / ("budget-" + format(threshold, ".9g").replace(".", "p"))
                latest_checkpoint = save_predictor(model, tensorizer, online_config, curve_path / "checkpoint.pt", records=records, optimizer=optimizer)
                point = {"budget_threshold": threshold, "actual_cost": ledger.spent, "experiments": iteration,
                         "records": len(records), "updates": len(updates), "checkpoint": latest_checkpoint,
                         "new_associations": len(new_associations),
                         "discoveries_per_execution_cost": len(new_associations) / max(ledger.spent, 1e-6)}
                if query_tasks:
                    if last_query_update != len(updates):
                        last_query = _frozen_queries(config, sorted(query_tasks), predict, curve_path / "queries")
                        last_query_update = len(updates)
                        query_cost += float(last_query.get("cost", 0))
                        query_execution_cost += float(last_query.get("environment_execution_cost", 0))
                    else:
                        point["query_reused_unchanged_predictor"] = True
                    point["frozen_query"] = copy.deepcopy(last_query)
                curves.append(point)
            # Save after each experiment so interrupted server runs retain evidence.
            write_jsonl(output / "transitions.jsonl", records)
            (output / "selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2))
        if records and iteration % int(options.get("update_every", 16)):
            update = update_predictor(model, tensorizer, replay.sample(int(options.get("replay_batch_size", 64))), online_config, optimizer=optimizer)
            optimizer = update.pop("optimizer", optimizer)
            updates.append({"iteration": iteration, "final_partial_period": True, **update})
        if records:
            latest_checkpoint = save_predictor(model, tensorizer, online_config, output / "checkpoint.pt", records=records, optimizer=optimizer)
            if query_tasks and last_query_update != len(updates):
                last_query = _frozen_queries(config, sorted(query_tasks), predict, output / "final-queries")
                last_query_update = len(updates)
                query_cost += float(last_query.get("cost", 0))
                query_execution_cost += float(last_query.get("environment_execution_cost", 0))
                curves.append({"actual_cost": ledger.spent, "final": True, "updates": len(updates), "frozen_query": last_query, "checkpoint": latest_checkpoint})
    except BudgetExceeded:
        stop = "budget"
    except Exception as error:
        stop = type(error).__name__ + ": " + str(error)
        raise
    finally:
        summary = {"strategy": options.get("strategy", "risk"), "cost": ledger.spent, "budget": ledger.limit,
                   "experiments": iteration, "records": len(records), "updates": updates, "learning_curve": curves,
                   "selections": selections, "budget_events": ledger.events, "stop_reason": stop,
                   "query_task_ids": sorted(query_tasks), "query_results_used_for_training": False,
                   "evidence_origins": sorted({r["evidence_origin"] for r in records}),
                   "model_calls": model_calls, "frozen_query_total_calls": query_cost,
                   "frozen_query_environment_execution_cost": query_execution_cost,
                   "budget_unit": "real_environment_execution_calls", "checkpoint": latest_checkpoint,
                   "total_environment_execution_cost": ledger.spent + query_execution_cost,
                   "new_associations": [list(a) for a in sorted(new_associations)],
                   "discoveries_per_execution_cost": len(new_associations) / max(ledger.spent, 1e-6),
                   "independent_repeat_confirmation_rate": len(confirmations) / len(new_associations) if new_associations else None}
        summary.update(run_context)
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary
