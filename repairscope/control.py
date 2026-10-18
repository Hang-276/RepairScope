"""Goal-aware execution, legal checking and repair with a single predictor."""
from __future__ import annotations

import copy
import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from .actor import normalize_action, noop_action, public_goals, public_history, public_state
from .candidates import predict_and_rank


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetLedger:
    limit: float
    spent: float = 0.0
    events: list[dict] = field(default_factory=list)

    @property
    def remaining(self):
        return max(0.0, self.limit - self.spent)

    def charge(self, kind: str, units: float = 1.0, **detail):
        units = float(units)
        if not math.isfinite(units) or units < 0:
            raise ValueError("Budget units must be finite and nonnegative")
        if units > self.remaining + 1e-9:
            raise BudgetExceeded(f"Budget exhausted before {kind}: {units} > {self.remaining}")
        self.spent += units
        event = {"kind": kind, "units": units, "cumulative_cost": self.spent, **detail}
        self.events.append(event)
        return event

    def call(self, kind: str, fn: Callable, units: float = 1.0, **detail):
        event = self.charge(kind, units, **detail)
        start = time.monotonic()
        try:
            result = fn()
            event["valid"] = True
            return result
        except Exception as error:
            event.update(valid=False, error=type(error).__name__)
            raise
        finally:
            event["latency_seconds"] = time.monotonic() - start

    def settle(self, kind: str, actual: float, reserved: float, **detail):
        """Record all incurred work even if an atomic backend call exceeds its bound."""
        difference = float(actual) - float(reserved)
        if difference:
            self.spent += difference
            self.events.append({"kind": kind, "units": difference, "cumulative_cost": self.spent, **detail})
        if self.spent > self.limit + 1e-9:
            raise BudgetExceeded("Backend exceeded its reserved atomic-call budget; incurred work was recorded")


def evaluate_goals(state: dict, goals: list[dict], previous: dict | None = None) -> list[bool | None]:
    from .schema import evaluate_goals as evaluate_predicates
    return evaluate_predicates(state, goals, previous=previous)


def goal_transition(before, after):
    return [None if a is None or b is None else 2 * int(a) + int(b) for a, b in zip(before, after)]


def protected_mask(goals: list[dict], before: list, step: int) -> list[bool]:
    # An ordinary subgoal becomes protected only after it has actually been satisfied.
    from .schema import effective_protected
    return [before[i] is True and effective_protected(g, step, before[i]) for i, g in enumerate(goals)]


def score_prediction(state, prediction, goals, action, step=0, config=None):
    config = config or {}
    before = evaluate_goals(state, goals)
    after = evaluate_goals(prediction.get("next_state", state), goals, previous=state)
    protection = protected_mask(goals, before, step)
    progress, damage, unknown = 0.0, 0.0, 0.0
    probs = prediction.get("goal_probabilities", [])
    for i, goal in enumerate(goals):
        weight = float(goal.get("weight", 1.0))
        if before[i] is not None and after[i] is not None:
            progress += weight * (int(after[i]) - int(before[i]))
            damage += weight * int(protection[i] and after[i] is False)
        elif config.get("use_goal_head", True) and i < len(probs) and len(probs[i]) == 4:
            progress += weight * (float(probs[i][1]) - float(probs[i][2]))
            damage += weight * float(probs[i][2]) * protection[i]
            unknown += weight
        else:
            unknown += weight
    risk_enabled = config.get("use_escape", float(config.get("escape_weight", 1)) > 0)
    risk = float(prediction.get("goal_escape", prediction.get("escape", 0.0))) if risk_enabled else 0.0
    uncertainty = float(prediction.get("uncertainty", 0.0))
    score = (float(config.get("progress_weight", 1)) * progress -
             float(config.get("damage_weight", 4)) * damage -
             float(config.get("risk_weight", config.get("escape_weight", 1))) * risk -
             float(config.get("uncertainty_weight", 0.1)) * uncertainty -
             float(config.get("unknown_weight", 0.1)) * unknown -
             float(config.get("cost_weight", 0.02)) * float(action.get("cost", 1)) -
             float(config.get("scope_weight", 0.0001)) * len(prediction.get("scope", [])))
    return {"score": score, "progress": progress, "damage": damage, "escape": risk,
            "uncertainty": uncertainty, "goal_before_public": before, "goal_after_public": after}


class RepairController:
    def __init__(self, actor, predictor: Callable | None, config: dict | None = None,
                 budget: BudgetLedger | None = None):
        self.actor, self.predictor = actor, predictor
        self.config = config or {}
        self.budget = budget or BudgetLedger(float(self.config.get("budget", self.config.get("max_calls", 100))))
        self.logs: list[dict] = []
        self.history: list[dict] = []
        self.repairs = 0
        self.checks = 0

    def _predict(self, state, action, goals, step):
        record = {"state": public_state(state), "action": normalize_action(action),
                  "history": public_history(self.history), "goals": public_goals(goals), "step": step}
        if self.predictor is None:
            return {"next_state": public_state(state), "scope": [], "escape": 0.0}
        return self.budget.call("prediction", lambda: self.predictor(record),
                                float(self.config.get("prediction_cost", 1)))

    def rank(self, state, goals, candidates, step=0):
        return predict_and_rank([normalize_action(a) for a in candidates],
            predict=lambda action: self._predict(state, action, goals, step),
            score=lambda action, prediction: score_prediction(state, prediction, goals, action, step, self.config))

    def _candidates(self, state, goals, step, mode, instruction):
        count = int(self.config.get("candidate_count", 4))
        candidates = self.budget.call("actor", lambda: self.actor.generate(
            public_state(state), public_goals(goals), public_history(self.history), count=count,
            mode=mode, instruction=instruction), float(self.config.get("actor_cost", 1)), mode=mode)
        # Every extra HTTP attempt (format retry) is a real policy call and is
        # charged at the same actor unit, never hidden as a free retry.
        extra = max(0, int(getattr(self.actor, "last_http_calls", 0)) - 1)
        if extra:
            self.budget.settle("actor_retry_adjustment", float(extra), 0.0, mode=mode)
        if mode == "repair":
            # Reserve a slot for no repair; do not silently expand the common candidate budget.
            candidates = [a for a in candidates if a.get("name") != "noop"][:max(0, count - 1)]
            candidates.append(noop_action())
        return candidates[:count]

    def _execute(self, env, state, action, selected, step, phase):
        action = normalize_action(action)
        if action["name"] == "noop":
            result = self.budget.call("noop", lambda: {"state": public_state(state), "cost": action["cost"],
                   "valid": True, "done": False}, action["cost"], phase=phase)
        else:
            if hasattr(env, "set_budget_remaining"):
                env.set_budget_remaining(self.budget.remaining)
            try:
                result = self.budget.call("environment", lambda: env.execute(action), action["cost"], phase=phase)
            except BudgetExceeded:
                raise
            except Exception as error:
                charged = getattr(error, "charged_cost", None)
                if charged is None:
                    match = re.search(r"charged_cost[=:]\s*([0-9.]+)", str(error))
                    charged = float(match.group(1)) if match else None
                if charged is not None:
                    self.budget.settle("failed_environment_actual_adjustment", charged, action["cost"], phase=phase)
                else:
                    self.budget.events[-1]["cost_uncertain"] = True
                raise
            actual_cost = float(result.get("cost", action["cost"]))
            self.budget.settle("environment_actual_adjustment", actual_cost, action["cost"], phase=phase)
        observed = result.get("state")
        if observed is None:
            observed = self.budget.call("observation", env.observe,
                                         float(self.config.get("observation_cost", 1)))
        observed = public_state(observed)
        log = {"step": step, "phase": phase, "state": public_state(state), "action": action,
               "next_state": observed, "valid": result.get("valid", True),
               "error": result.get("error"), "prediction": copy.deepcopy(selected.get("prediction", {})),
               "scope": list(selected.get("prediction", {}).get("scope", [])),
               "history": public_history(self.history), "actual_cost": result.get("cost", action["cost"]),
               "cumulative_cost": self.budget.spent}
        # Audit is evaluation/training output only; it is never used to build history or prompts.
        if "audit" in result:
            log["audit"] = copy.deepcopy(result["audit"])
        self.logs.append(log)
        self.history.append({"state": public_state(state), "action": action})
        return observed, result

    def _legal_check(self, env, state, goals, selected, step):
        threshold = float(self.config.get("risk_threshold", 0.5))
        risk_enabled = self.config.get("use_escape", float(self.config.get("escape_weight", 1)) > 0)
        check_risk = (float(selected.get("escape", 0)) if risk_enabled else
                      float(selected.get("uncertainty", 0)))
        if check_risk < threshold or self.checks >= int(self.config.get("max_checks", 4)):
            return state, False
        actions = getattr(env, "metadata", {}).get("read_only_actions", [])
        # Only a backend-supplied legal read allowlist may be executed as a check.
        if not actions:
            return state, False
        action = normalize_action(actions[0])
        if not action.get("read_only", False):
            raise ValueError("Environment check action is not explicitly read_only")
        next_state, _ = self._execute(env, state, action, {"prediction": {}}, step, "check")
        self.checks += 1
        return next_state, True

    def run(self, env, goals=None, instruction="", reset=True):
        state = (self.budget.call("reset", env.reset, float(self.config.get("reset_cost", 1))) if reset else
                 self.budget.call("observation", env.observe, float(self.config.get("observation_cost", 1))))
        state = public_state(state)
        metadata = getattr(env, "metadata", {})
        goals = public_goals(goals if goals is not None else metadata.get("goals", []))
        instruction = instruction or metadata.get("instruction", "")
        if metadata.get("api_docs"):
            from .actor import actor_instruction
            # Compact public catalog; the full per-response schemas overflow any
            # candidate context and remain available through legal API queries.
            instruction = actor_instruction(metadata)
        stop_reason, errors = "max_steps", []
        for step in range(int(self.config.get("max_steps", 30))):
            try:
                before = evaluate_goals(state, goals)
                candidates = self._candidates(state, goals, step, "execute", instruction)
                if self.predictor is None:
                    ranked = [{"action": candidates[0], "prediction": {}, "escape": 0, "score": 0}]
                else:
                    ranked = self.rank(state, goals, candidates, step)
                selected = ranked[0]
                state, checked = self._legal_check(env, state, goals, selected, step)
                if checked:
                    ranked = self.rank(state, goals, candidates, step)
                    selected = ranked[0]
                    before = evaluate_goals(state, goals)
                previous_state = state
                state, result = self._execute(env, state, selected["action"], selected, step, "execute")
                after = evaluate_goals(state, goals, previous=previous_state)
                self.logs[-1].update(goals=goals, goal_before=before, goal_after=after,
                                     goal_transitions=goal_transition(before, after), candidates=ranked)
                damage = any(p and b is False for p, b in zip(protected_mask(goals, before, step), after))
                predicted = selected.get("prediction", {}).get("next_state")
                mismatch = predicted is not None and evaluate_goals(predicted, goals, previous=previous_state) != after
                if (damage or mismatch) and self.config.get("repair", True) and self.predictor is not None and self.repairs < int(self.config.get("max_repairs", 4)):
                    repair_before = after
                    repair_previous = state
                    repairs = self._candidates(state, goals, step, "repair", instruction)
                    repair_ranked = self.rank(state, goals, repairs, step)
                    state, repair_result = self._execute(env, state, repair_ranked[0]["action"], repair_ranked[0], step, "repair")
                    repaired = evaluate_goals(state, goals, previous=repair_previous)
                    self.logs[-1].update(goals=goals, goal_before=repair_before, goal_after=repaired,
                                         goal_transitions=goal_transition(repair_before, repaired), candidates=repair_ranked)
                    self.repairs += 1
                    result = repair_result if repair_result.get("done") else result
                if result.get("done") or selected["action"]["name"] in {"stop", "finish"}:
                    stop_reason = "done"
                    break
            except BudgetExceeded:
                stop_reason = "budget"
                break
            except (RuntimeError, ValueError, KeyError, OSError) as error:
                errors.append({"step": step, "type": type(error).__name__, "message": str(error)})
                stop_reason = "execution_error"
                break
        return {"state": state, "goals": goals, "logs": self.logs, "cost": self.budget.spent,
                "budget": {"limit": self.budget.limit, "spent": self.budget.spent, "events": self.budget.events},
                "checks": self.checks, "repairs": self.repairs, "stop_reason": stop_reason,
                "budget_overrun": max(0.0, self.budget.spent - self.budget.limit), "errors": errors}
