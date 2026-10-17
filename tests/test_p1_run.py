"""Exercise P1 orchestration end to end with explicitly synthetic execution.

Only P0 loading, model transport and environment execution are replaced. Probe
validation, revision parsing, budget ledgers, commit validation, immutable store,
transfer generation and report serialization all use production implementations.
"""

from copy import deepcopy
import json
from types import SimpleNamespace

import repairscope.p1 as p1
from repairscope.protocol import digest
from repairscope.knowledge import RuleStore

from test_p1_contract import frozen_inputs


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class SyntheticModelClient:
    def __init__(self):
        self.calls = []
        self.plans = 0

    def complete(self, messages, max_tokens, budget, purpose):
        reservation = budget.reserve_chat(10, max_tokens, purpose)
        system = messages[0]["content"]
        data = json.loads(messages[1]["content"])
        if system.startswith("You propose bounded operation"):
            self.plans += 1
            result = {
                "mechanism_id": f"synthetic_dependency_mechanism_{self.plans}",
                "r0": "flag = 0", "r1": "flag = 1",
                "conditions": [None, {"code": "apis.fixture.update_condition(value=True)",
                                      "query": "print(apis.fixture.get_condition())", "expected_output": "True\n",
                                      "public_api_only": True, "semantics_preserved": True, "checker_invariant": True}],
                "explanations": ["compensation is conditional", "compensation is always useful"],
            }
        elif system.startswith("You propose one conditional repair"):
            snapshot = data["knowledge_snapshot"]
            evidence_id = data["offline_execution_evidence"][0]["evidence_id"]
            previous = snapshot["rules"][0] if snapshot["rules"] else None
            result = {
                "rule_id": "parameterized_dependency_repair", "version": 1 if previous is None else previous["version"] + 1,
                "intent": "conditionally preserve existing progress",
                "applicability": {"observable_conditions": [{"key": "condition", "equals": True}],
                                  "required_queries": [], "unknown_policy": "do_not_treat_as_verified"},
                "repair": {"base_template": "repair the requested write", "conditional_compensation": [],
                           "read_scope": ["public dependency"], "write_scope": ["requested state"]},
                "goal_effects": {"recovery_claims": ["repair requested state"], "preservation_claims": ["keep prior progress"]},
                "evidence_ids": ([] if previous is None else previous["evidence_ids"]) + [evidence_id],
                "counterexample_ids": [], "status": "validated",
            }
            if previous is not None:
                result["applicability"]["observable_conditions"].append({"key": "visible_scope_ok", "equals": True})
                result["repair"]["conditional_compensation"] = ["apply compensation inside the observed scope"]
        elif system.startswith("Generate a current-state repair"):
            injected = json.loads(system.splitlines()[-1])
            version = max((rule["version"] for rule in injected), default=0)
            result = {"program": f"flag = {22 if version >= 2 else 11}"}
        else:
            raise AssertionError(f"Unexpected model request: {system[:80]}")
        self.calls.append({"messages": deepcopy(messages), "reply": deepcopy(result), "purpose": purpose})
        budget.record_chat(reservation, input_tokens=10, output_tokens=10, seconds=0.001)
        return SimpleNamespace(content=json.dumps(result), output_tokens=10, seconds=0.001)


def synthetic_replay(request):
    # Deliberately claims AppWorld here: run_p1 must override injected adapters to
    # synthetic_fixture so this test can never establish a scientific P1 pass.
    program = request["program"]
    changed_condition = request.get("intervention") is not None
    failure = program == "flag = -1" or (request["task_id"] == "fixture_task_3" and program == "flag = 11")
    keep = not (changed_condition and program == "flag = 0")
    before = {"fix": False, "keep": True}
    return {
        "ok": True, "errors": [], "program_error": None, "checker_valid": True,
        "prefix_verified": True, "intervention_verified": True,
        "task_id": request["task_id"], "scenario_id": request["scenario_id"], "split": request["split"],
        "seed": request.get("seed", 100), "identity": deepcopy(request["identity"]),
        "g_fix": request["g_fix"], "g_keep": request["g_keep"], "goal_mapping": deepcopy(request["goal_mapping"]),
        "prefix_fingerprint": digest([request["task_id"], "fixed-prefix"]),
        "goals_initial": deepcopy(before), "goals_before": deepcopy(before),
        "goals_after": {"fix": not failure, "keep": keep},
        "before_fingerprint": digest([request["task_id"], changed_condition]),
        "after_fingerprint": digest([request["task_id"], changed_condition, program]),
        "query_output": "True\n" if changed_condition else "False\n",
        "evidence_origin": "AppWorld", "source": request["source"], "program": program,
        "program_sha256": digest(program),
    }


def _run_study(tmp_path, monkeypatch, *, exercise_budget=False):
    document, assets, protocol, manifest, goals = frozen_inputs()
    mapping = {"fix": {"requirements": ["synthetic fix requirement"], "complete": True},
               "keep": {"requirements": ["synthetic keep requirement"], "complete": True}}
    for event in document["events"]:
        event["request"]["goal_mapping"] = deepcopy(mapping)
        event["request"]["identity"]["mapping_sha256"] = digest(mapping)
        event["public_context"]["observations"] = {"condition": True, "visible_scope_ok": True}
        event["candidates"] = [{"id": "retained-failure", "code": "flag = -1"},
                               {"id": "existing-success", "code": "flag = 1"}]
    # The third original task has only a failed candidate and stays unresolved.
    document["events"][2]["candidates"] = [{"id": "only-failure", "code": "flag = -1"}]
    if exercise_budget:
        document["events"][2]["candidates"] = [
            {"id": f"existing-{index}", "code": f"flag = {value}"}
            for index, value in enumerate((-1, 1, 33, 44))]
    case = document["transfer_cases"][0]
    case["request"]["goal_mapping"] = deepcopy(mapping)
    case["request"]["identity"]["mapping_sha256"] = digest(mapping)
    case["public_context"]["observations"] = {"condition": True, "visible_scope_ok": True}
    inputs = tmp_path / "synthetic_events.json"
    inputs.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(p1, "load_p0", lambda directory: (assets, protocol, manifest, goals))
    client = SyntheticModelClient()
    requests = []
    def replay(request):
        requests.append(deepcopy(request))
        return synthetic_replay(request)
    output = tmp_path / "synthetic_p1"
    report = p1.run_p1(tmp_path / "unused_p0", inputs, output, "http://fixture.invalid", client=client, replay=replay)
    return document, report, output, requests, client


def test_two_stage_revision_holdout_and_failure_records_are_integrated(tmp_path, monkeypatch):
    document, report, output, requests, client = _run_study(tmp_path, monkeypatch)
    witnesses = read_rows(output / "witnesses.jsonl")
    updates = read_rows(output / "knowledge_versions.jsonl")
    transfers = read_rows(output / "transfer_cases.jsonl")
    events = read_rows(output / "events.jsonl")
    final = RuleStore(output / "knowledge").snapshot()
    assert len(witnesses) == 5 and sum(w["recovery_success"] is False for w in witnesses) == 3
    assert len(updates) == 2 and all(update["validation"]["accepted"] for update in updates), updates
    assert updates[0]["is_revision"] is False and updates[1]["is_revision"] is True
    assert final["version"] == 2 and final["rules"][0]["version"] == 2
    assert len(final["rules"][0]["evidence_ids"]) == 2
    assert updates[1]["fixed_program_regression"][0]["passed"] is True
    assert len(transfers) == 1 and transfers[0]["improved"] is True
    assert transfers[0]["knowledge_changed"] is True and transfers[0]["source"] == "synthetic_fixture"
    assert transfers[0]["old"]["success"] is False and transfers[0]["new"]["success"] is True
    assert [event["learning_usage"]["calls"] for event in events] == [3, 4, 0]
    assert [event["new_branches"] for event in events] == [7, 9, 1]
    assert report["branches"] == {"initial": 17, "extension": 0}
    assert len(requests) == 19  # 17 learning branches + separately budgeted old/new holdout.
    assert events[2]["status"] == "unresolved"
    assert [event["original_score"] for event in events] == [event["original_score"] for event in document["events"]]
    assert all(witness["evidence_origin"] == "synthetic_fixture" for witness in witnesses)
    assert report["evidence_origin"] == "synthetic_fixture"
    assert report["candidate_attempts"] == 5
    assert not report["A"] and not report["B"] and not report["C"]
    assert report["decision"] == "stop_or_analyze_existing_evidence" and report["p2_started"] is False
    # Transfer contents were not sent to any proposal/revision call.
    for call in client.calls:
        if call["messages"][0]["content"].startswith("You propose"):
            assert "fixture_task_3" not in call["messages"][1]["content"]


def test_branch_exhaustion_preserves_uncommitted_rule_and_partial_regression(tmp_path, monkeypatch):
    document, report, output, requests, client = _run_study(tmp_path, monkeypatch, exercise_budget=True)
    updates = read_rows(output / "knowledge_versions.jsonl")
    events = read_rows(output / "events.jsonl")
    assert len(updates) == 3
    exhausted = updates[-1]
    assert exhausted["rule"]["version"] == 3
    assert exhausted["validation"]["accepted"] is False
    assert any("BudgetExceeded" in reason for reason in exhausted["validation"]["reasons"])
    assert exhausted["current_regenerated_behavior"]["receipt"]["ok"] is True
    assert len(exhausted["fixed_program_regression"]) == 2
    assert exhausted["fixed_program_regression"][0]["passed"] is True
    assert exhausted["fixed_program_regression"][1]["known_harm"] is None
    assert "regenerated_receipt" not in exhausted["fixed_program_regression"][1]
    assert RuleStore(output / "knowledge").snapshot()["version"] == 2
    assert [event["new_branches"] for event in events] == [7, 9, 12]
    assert [event["learning_usage"]["calls"] for event in events] == [3, 4, 5]
    assert report["branches"]["initial"] == 28
    assert len(requests) == 30  # Failed reserve adds no unaccounted execution.
    assert report["p2_started"] is False
