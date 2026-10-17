"""Bounded one-shot mechanism study; no continuous learner or test entry point."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from .budget import Budget, BudgetExceeded
from .io import atomic_json, write_jsonl
from .knowledge import RuleStore, rule_prompt, validate_rule
from .llm import ChatClient, LLMError, count_tokens, strict_json_loads
from .logging import RunLog
from .clock import elapsed_time
from .p0 import load_p0
from .probes import run_joint_probe
from .protocol import BranchBudget, NATURAL, check_human_edits, digest, read_json, require_train
from .replay import run_branch, receipt_success
from .revise import propose_revision, _check_public
from .validate import validate_update

PUBLIC_FIELDS = {"instruction", "history", "api_docs", "observations"}


def public_view(value):
    if not isinstance(value, dict) or set(value) - PUBLIC_FIELDS:
        raise ValueError("Public model context accepts only instruction/history/api_docs/observations")
    if not isinstance(value.get("instruction"), str) or not value["instruction"].strip():
        raise ValueError("Original public instruction required")
    _check_public(value)
    if "history" in value:
        if not isinstance(value["history"], list) or any(
                not isinstance(m, dict) or set(m) != {"role", "content"}
                or m["role"] not in {"system", "user", "assistant", "tool"}
                or not isinstance(m["content"], str) for m in value["history"]):
            raise ValueError("Public history contains only visible role/content messages")
    return deepcopy(value)


def _code(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Model must return a nonempty fixed Python program")
    compile(value, "<model-repair>", "exec")
    return value


class EventModel:
    """Every request, including failed/truncated requests, uses one shared ledger."""
    def __init__(self, client, ledger, log=None):
        self.client, self.ledger, self.log = client, ledger, log

    def __call__(self, messages, max_tokens=3000):
        if self.log:
            self.log.info(f"Model request {self.ledger.calls + 1}/{self.ledger.max_calls} | "
                          f"output budget remaining: {self.ledger.remaining_output_tokens:,} tokens")
        result = self.client.complete(messages, max_tokens=min(max_tokens, 3000),
                                      budget=self.ledger, purpose="offline_learning")
        if self.log:
            self.log.info(f"Model response received | {result.output_tokens:,} output tokens | {result.seconds:.2f}s")
        return strict_json_loads(result.content)


def propose_joint(model, context, snapshot, witness):
    # Checker truth is allowed in offline learning, never in online candidate prompts.
    prompt = {
        "public_context": context, "knowledge": snapshot,
        "executed_witness": witness,
        "required_output": {"mechanism_id": "parameterized_relation_type", "r0": "fixed Python",
                            "r1": "fixed Python with compensation at its semantic position",
                            "conditions": [None, {"code": "public API intervention", "query": "print observable condition",
                                           "expected_output": "exact observation", "public_api_only": True,
                                           "semantics_preserved": True, "checker_invariant": True}],
                            "explanations": ["at most three competing explanations"]},
    }
    return model([
        {"role": "system", "content": "You propose bounded operation x condition experiments. Return JSON only. "
         "Use supplied public API docs. Preserve task semantics, G_fix and G_keep. No task-specific hand-written answers, "
         "DB access, file access, solution access, or invented APIs. A failed program is not necessity evidence. "
         "Use one fixed r0 and one r1 across both conditions; insert compensation where required. "
         "Return an empty object if no legal observable intervention is supported. "
         "Your flags are proposals only; execution and an independently audited checker validate them."},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}])


def _applicable_rules(context, snapshot):
    # Retrieval uses observations only. No truth vector, mapping or receipt is sent.
    observations = context.get("observations", {})
    if not isinstance(observations, dict):
        raise ValueError("Visible observations must be a key/value object")
    # Read snapshots through a temporary in-memory store without mutating the source.
    return sorted([rule for rule in snapshot["rules"] if rule.get("status") in {"validated", "restricted"}
             and all(c.get("key") in observations and observations[c["key"]] is not None
                     and digest(observations[c["key"]]) == digest(c.get("equals"))
                     for c in rule["applicability"]["observable_conditions"])], key=lambda r: r["rule_id"])[:3]


def _repair_with_knowledge(model, context, snapshot):
    rules = _applicable_rules(context, snapshot)
    injected = rule_prompt(rules)
    if count_tokens(injected) > 2048:
        raise ValueError("Knowledge injection exceeds conservative 2048 token limit")
    answer = model([
        {"role": "system", "content": "Generate a current-state repair using visible facts and applicable knowledge. "
         "Preserve the entire original instruction. Unknown conditions are not verified. Return JSON {\"program\":\"Python\"}.\n" + injected},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)}])
    return _code(answer["program"])


def _event_identity(event):
    request = event["request"]
    return digest({"task_id": request["task_id"], "steps": request["steps"],
                   "identity": request["identity"]})


def _validate_inputs(document, assets, protocol, manifest, goals):
    if set(document) - {"schema_version", "events", "transfer_cases", "allow_extension"}:
        raise ValueError("Unknown P1 input fields")
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    events = document["events"]
    if not events:
        raise ValueError("No existing train events supplied")
    ids, states = set(), set()
    phase_states = {"initial": 0, "extension": 0}
    rank = {t: i for i, t in enumerate(protocol["task_order"])}
    task_checker_locks = {}
    last_rank, last_phase = -1, "initial"
    def check_frozen(request, context):
        task_id = request["task_id"]
        frozen = goals["tasks"][task_id]
        if request["scenario_id"] != tasks[task_id]["scenario_id"]:
            raise ValueError("Scenario identity differs from the frozen task manifest")
        if public_view(context)["instruction"] != frozen["instruction"]:
            raise ValueError("Public instruction differs from frozen D1")
        expected = {"protocol_sha256": digest(protocol), "goals_sha256": digest(frozen),
                    "environment_sha256": assets["environment_sha256"],
                    "task_sha256": tasks[task_id]["task_sha256"],
                    "mapping_sha256": digest(request["goal_mapping"])}
        if request["identity"] != expected:
            raise ValueError("Event identity does not match frozen P0 assets/mapping")
        if set(request["g_fix"]) & set(request["g_keep"]) or not request["g_fix"] or not request["g_keep"]:
            raise ValueError("Disjoint nonempty G_fix and G_keep are declared before execution")
        goal_ids = {g["id"] for g in frozen["goals"]}
        if set(request["g_fix"] + request["g_keep"]) - goal_ids or set(request["goal_mapping"]) != goal_ids:
            raise ValueError("Repair promise/mapping refers to an unfrozen goal")
        if request.get("intervention") is not None:
            raise ValueError("Natural event and transfer prefixes cannot already contain a condition intervention")
        checker_lock = digest({"mapping": request["goal_mapping"], "checker_spec": request.get("checker_spec"),
                               "audit": request.get("checker_audit_sha256")})
        if task_id in task_checker_locks and task_checker_locks[task_id] != checker_lock:
            raise ValueError("A task's frozen checker or mapping cannot vary across natural states")
        task_checker_locks[task_id] = checker_lock
    for event in events:
        request = event["request"]
        require_train(request["split"])
        if request["task_id"] not in tasks:
            raise ValueError("Event task is outside the frozen train allowlist")
        if event["event_id"] in ids or _event_identity(event) in states:
            raise ValueError("Duplicate event/state would inflate the denominator")
        ids.add(event["event_id"]); states.add(_event_identity(event))
        phase = event.get("phase", "initial")
        if phase not in phase_states or (last_phase == "extension" and phase == "initial"):
            raise ValueError("Invalid P1 phase order")
        if phase == "extension" and document.get("allow_extension") is not True:
            raise ValueError("Extension must be predeclared before any P1 results")
        if rank[request["task_id"]] < last_rank:
            raise ValueError("Events must follow the frozen scenario round-robin task order")
        last_rank, last_phase = rank[request["task_id"]], phase
        phase_states[phase] += 1
        check_human_edits(event)
        if request["source"] not in NATURAL | {"natural_incomplete"}:
            raise ValueError("P1 starts from native natural train states")
        if request.get("factory_spec"):
            raise ValueError("Custom factories are fixture-only; live P1 rejects them")
        if not 1 <= len(event["candidates"]) <= 4:
            raise ValueError("Each state must retain its original 1..4 candidate pool")
        if any(not isinstance(c, dict) or not isinstance(c.get("code"), str) for c in event["candidates"]):
            raise ValueError("Every existing candidate must retain its code string, including invalid syntax")
        check_frozen(request, event["public_context"])
        score = event["original_score"]
        if not (score.get("scored") is True and score.get("task_id") == request["task_id"]
                and score.get("split") == "train" and "result" in score
                and score.get("result_sha256") == digest(score["result"])):
            raise ValueError("Original task must already be scored with an immutable score hash")
        if not event.get("forbidden_literals") or request["task_id"] not in event["forbidden_literals"]:
            raise ValueError("Provide audit-only task/entity/answer literals to exclude from deployable rules")
    if any(n > 15 for n in phase_states.values()):
        raise ValueError("Each preset phase has at most 15 natural states")
    initial_tasks = {e["request"]["task_id"] for e in events if e.get("phase", "initial") == "initial"}
    extension_tasks = {e["request"]["task_id"] for e in events if e.get("phase") == "extension"}
    if initial_tasks & extension_tasks:
        raise ValueError("The one extension may use only previously unexamined train tasks")
    if not 3 <= len(initial_tasks) <= 5:
        raise ValueError("Initial P1 coverage must contain 3..5 original train tasks")
    cases = document.get("transfer_cases", [])
    if len(cases) > 3 or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("At most three unique independent transfer cases")
    touched = {e["request"]["task_id"] for e in events}
    for case in cases:
        request = case["request"]
        require_train(request["split"])
        if request["task_id"] in touched:
            raise ValueError("Transfer task participated in induction/selection/regression; not independent")
        if request["task_id"] not in tasks or request.get("factory_spec"):
            raise ValueError("Transfer must use existing real train holdout tasks")
        if request.get("source") not in NATURAL | {"natural_incomplete"}:
            raise ValueError("Transfer must originate from natural train states")
        if type(case.get("repeats", 1)) is not int or not 1 <= case.get("repeats", 1) <= 3:
            raise ValueError("At most three old/new paired behavior repeats")
        check_frozen(request, case["public_context"])
        check_human_edits(case)
    return events, cases


def run_p1(p0_dir, events_file, output_dir, base_url, *, client=None, replay=run_branch):
    assets, protocol, manifest, goals = load_p0(p0_dir)
    document = read_json(events_file)
    events, transfer_inputs = _validate_inputs(document, assets, protocol, manifest, goals)
    output = Path(output_dir)
    if output.exists():
        raise ValueError("P1 output already exists; do not rerun until success or overwrite scores")
    output.mkdir(parents=True)
    log = RunLog(output, "P1")
    wall_started = elapsed_time()
    replay_costs = []
    log.info("Mechanism study started. Model weights are frozen; only train evidence is used.")
    log.info(f"Plan: {len(events)} natural states | {len(transfer_inputs)} independent transfer cases | "
             "12 shadow branches, 6 model calls and 8,000 output tokens per learning event")
    # Freeze exact input before the first execution or model call.
    atomic_json(output / "inputs.lock.json", document)
    atomic_json(output / "protocol.lock.json", protocol)
    store = RuleStore(output / "knowledge")
    atomic_json(output / "K0.json", store.snapshot())
    client = client or ChatClient(base_url, protocol["model"]["name"], context_length=32000,
                                  max_retries=0, chat_template_kwargs={"enable_thinking": False})
    budgets = {phase: BranchBudget() for phase in ("initial", "extension")}
    witnesses, probes, updates, transfers, event_rows = [], [], [], [], []
    regression_archive, accepted_pairs = [], []
    used_forbidden = set()
    started = datetime.now(timezone.utc).isoformat()
    def save():
        for name, rows in (("witnesses", witnesses), ("joint_probes", probes),
                           ("knowledge_versions", updates), ("transfer_cases", transfers), ("events", event_rows)):
            write_jsonl(output / f"{name}.jsonl", rows)
    current_task, task_snapshot = None, None
    task_learning_events = {}
    actual_replay = replay
    def replay(req):
        receipt = actual_replay(req)
        if actual_replay is not run_branch:
            receipt["evidence_origin"] = "synthetic_fixture"
        replay_costs.append(receipt.get("wall_seconds", receipt.get("duration_seconds", 0)))
        log.info(f"Replay {'OK' if receipt.get('ok') else 'FAILED'} | "
                 f"{receipt.get('wall_seconds', receipt.get('duration_seconds', 0)):.2f}s wall | "
                 f"goal labels {'known' if receipt.get('checker_valid') else 'unknown'}")
        return receipt
    for index, event in enumerate(events, 1):
        request = event["request"]
        if current_task != request["task_id"]:
            current_task, task_snapshot = request["task_id"], store.snapshot()
        before = store.snapshot()
        phase, event_id = event.get("phase", "initial"), event["event_id"]
        ledger = Budget(max_calls=6, max_output_tokens=8000)
        model = EventModel(client, ledger, log)
        log.progress("Natural states", index, len(events), detail=f"task={request['task_id']} | phase={phase}")
        row = {"event_id": event_id, "task_id": request["task_id"], "scenario_id": request["scenario_id"],
               "source": request["source"], "phase": phase, "status": "unresolved",
               "original_score": event["original_score"], "task_knowledge_sha256": digest(task_snapshot),
               "human_task_edits": 0, "human_rule_edits": 0, "human_intervention_edits": 0,
               "audit_minutes": event.get("audit_minutes", 0)}
        candidate_update = None
        used_forbidden.update(event["forbidden_literals"])
        def execute(req):
            budgets[phase].reserve(event_id)
            return replay(req)
        try:
            results = []
            for index, candidate in enumerate(event["candidates"]):
                fixed = {**deepcopy(request), "program": candidate.get("code"), "intervention": None}
                receipt = execute(fixed)
                receipt.update(event_id=event_id, candidate_id=candidate.get("id", str(index)),
                               recovery_success=receipt_success(receipt))
                witnesses.append(receipt); results.append((fixed, receipt))
                log.info(f"Candidate {index + 1}/{len(event['candidates'])}: "
                         f"{'RECOVERY VERIFIED' if receipt['recovery_success'] else 'FAILED OR UNKNOWN'}")
            useful = [pair for pair in results if pair[1]["recovery_success"]]
            if not useful:
                row["reason"] = "No successful current-state repair witness; all candidates retained"
                continue
            if task_learning_events.get(current_task, 0) >= 2:
                row["reason"] = "Two learning events already attempted for this task; state witness results retained"
                continue
            task_learning_events[current_task] = task_learning_events.get(current_task, 0) + 1
            witness_request, witness = useful[0]
            plan = propose_joint(model, public_view(event["public_context"]), task_snapshot, witness)
            if not plan:
                row["reason"] = "No justified legal condition intervention proposed"
                continue
            joint = run_joint_probe(request, plan, run=replay,
                                    reserve=lambda: budgets[phase].reserve(event_id))
            joint["event_id"] = event_id
            probes.append(joint)
            log.info(f"Joint probe: {'VALID' if joint.get('valid') else 'INVALID OR UNKNOWN'} | "
                     f"conditional effect: {'observed' if joint.get('conditional_effect') else 'not established'}")
            if not joint.get("valid") or not joint.get("conditional_effect"):
                row["reason"] = "Joint execution did not establish valid sample-level conditional compensation"
                continue
            if not any(receipt_success(joint["cells"][key]) for key in ("c0r1", "c1r1")):
                row["reason"] = "Compensated program never fulfils the declared recovery/preservation promise"
                continue
            rule = propose_revision(model, before, public_view(event["public_context"]), [joint])
            candidate_update = {"event_id": event_id, "rule": rule,
                                "old_snapshot_sha256": digest(before), "original_task_scored": True,
                                "is_revision": any(r["rule_id"] == rule["rule_id"] for r in before["rules"]),
                                "fixed_program_regression": [],
                                "regression_scope": "previously_accepted_natural_witness_prefixes",
                                "validation": {"accepted": False, "reasons": ["validation_not_completed"]}}
            updates.append(candidate_update)
            validate_rule(rule, forbidden_literals=tuple(used_forbidden))
            prospective = deepcopy(before)
            prospective["rules"] = [r for r in prospective["rules"] if r["rule_id"] != rule["rule_id"]] + [rule]
            prospective["version"] += 1
            if not any(r["rule_id"] == rule["rule_id"] for r in _applicable_rules(event["public_context"], prospective)):
                row["reason"] = "Proposed knowledge is not observable/applicable to current behavior; retained as an uncommitted candidate"
                candidate_update["validation"] = {"accepted": False, "reasons": ["new_rule_not_applied"]}
                continue
            behavior_program = _repair_with_knowledge(model, public_view(event["public_context"]), prospective)
            behavior = execute({**deepcopy(request), "program": behavior_program, "intervention": None})
            candidate_update["current_regenerated_behavior"] = {"program": behavior_program, "receipt": behavior}
            evidence = [{"evidence_id": joint["evidence_id"], "executed": True, "valid": True,
                         "recovery_success": receipt_success(behavior), "known_harm": not receipt_success(behavior),
                         "source": joint.get("evidence_origin", "unknown"), "split": "train", "joint_probe": joint,
                         "regenerated_behavior": behavior, "program": behavior_program}]
            # Re-execute all retained successful old fixed programs. No regression
            # is silently omitted to squeeze a growing archive into a fixed budget.
            checks = candidate_update["fixed_program_regression"]
            for archived in regression_archive:
                check = execute(archived["request"])
                check_row = {"valid": False, "passed": False, "known_harm": None, "fixed_program_receipt": check}
                checks.append(check_row)
                new_program = _repair_with_knowledge(model, archived["public_context"], prospective)
                new_check = execute({**deepcopy(archived["request"]), "program": new_program})
                passed = receipt_success(check) and receipt_success(new_check)
                check_row.update({"passed": passed, "valid": check.get("ok") is True and new_check.get("ok") is True,
                               "known_harm": not passed, "fixed_program_receipt": check,
                               "regenerated_program": new_program, "regenerated_receipt": new_check})
            old_rule = next((r for r in before["rules"] if r["rule_id"] == rule["rule_id"]), None)
            context = {"task_scored": True, "frozen_before": digest(protocol), "frozen_after": digest(protocol),
                       "automatic": True, "regression_checks": checks,
                       "human_task_edits": 0, "human_rule_edits": 0, "human_intervention_edits": 0}
            decision = validate_update(old_rule, rule, evidence, context)
            update = candidate_update
            update["validation"] = decision
            if decision["accepted"]:
                after = store.commit(rule, decision)
                update["new_snapshot_sha256"] = digest(after)
                accepted_pairs.append({"before": before, "after": after,
                                       "is_revision": old_rule is not None, "event_id": event_id,
                                       "changed_rule_id": rule["rule_id"]})
                regression_archive.append({"request": witness_request, "public_context": public_view(event["public_context"])})
                row["status"] = "knowledge_committed"
                log.info(f"Knowledge committed | snapshot {after['version']} | rule version {rule['version']}")
            else:
                row["status"] = "update_rejected"
        except (BudgetExceeded, LLMError, ValueError, KeyError, TypeError, SyntaxError, OSError) as error:
            row["reason"] = f"{type(error).__name__}: {error}"
            if candidate_update is not None:
                candidate_update["validation"] = {"accepted": False, "reasons": [row["reason"]]}
        finally:
            row["learning_usage"] = ledger.snapshot()
            row["new_branches"] = budgets[phase].events.get(event_id, 0)
            event_rows.append(row)
            save()
            log.info(f"State result: {row['status'].replace('_', ' ').upper()} | "
                     f"branches {row['new_branches']}/12 | model calls {ledger.calls}/6 | "
                     f"output {ledger.output_tokens:,}/8,000 tokens")
            if row.get("reason"):
                log.warning(row["reason"] if row["reason"].isascii() else
                            "State validation was unresolved. Detailed evidence is retained in events.jsonl.")
    # Transfer is revealed after all induction/selection/regression has ended.
    # It never controls knowledge commit or returns to the updater.
    revisions = [pair for pair in accepted_pairs if pair["is_revision"]]
    if revisions:
        pair = revisions[-1]
        for case in transfer_inputs:
            for repeat in range(case.get("repeats", 1)):
                log.info(f"Independent transfer | case={case['case_id']} | pair {repeat + 1}/{case.get('repeats', 1)}")
                result = {"case_id": case["case_id"], "task_id": case["request"]["task_id"],
                          "repeat": repeat, "update_event": pair["event_id"], "independent": True,
                          "source": "unknown", "old": None, "new": None, "status": "unresolved"}
                behavior_budget = Budget(max_calls=2, max_output_tokens=6000)
                model = EventModel(client, behavior_budget, log)
                try:
                    for name, snapshot in (("old", pair["before"]), ("new", pair["after"])):
                        program = _repair_with_knowledge(model, public_view(case["public_context"]), snapshot)
                        receipt = replay({**deepcopy(case["request"]), "program": program, "intervention": None})
                        retrieved = _applicable_rules(case["public_context"], snapshot)
                        result[name] = {"program": program, "receipt": receipt, "success": receipt_success(receipt),
                                        "injected_rules": [{"rule_id": r["rule_id"], "version": r["version"]} for r in retrieved],
                                        "knowledge_prompt_sha256": digest(rule_prompt(retrieved))}
                    result["status"] = "evaluated"
                    result["source"] = "AppWorld" if all(result[n]["receipt"].get("evidence_origin") == "AppWorld" for n in ("old", "new")) else "synthetic_fixture"
                    result["behavior_changed"] = result["old"]["program"] != result["new"]["program"]
                    result["knowledge_changed"] = (result["old"]["knowledge_prompt_sha256"] != result["new"]["knowledge_prompt_sha256"]
                        and any(r["rule_id"] == pair["changed_rule_id"] for r in result["new"]["injected_rules"]))
                    result["improved"] = result["knowledge_changed"] and result["new"]["success"] and not result["old"]["success"]
                except (BudgetExceeded, LLMError, ValueError, KeyError, TypeError, SyntaxError, OSError) as error:
                    result["reason"] = f"{type(error).__name__}: {error}"
                result["usage"] = behavior_budget.snapshot()
                transfers.append(result)
                save()
                log.info(f"Transfer pair: {result['status'].upper()} | "
                         f"knowledge-dependent improvement: {'yes' if result.get('improved') else 'not established'}")
    atomic_json(output / "K_final.json", store.snapshot())
    report = summarize(witnesses, probes, updates, transfers, event_rows)
    all_usage = [r["learning_usage"] for r in event_rows] + [r["usage"] for r in transfers]
    report.update(started_at=started, completed_at=datetime.now(timezone.utc).isoformat(),
                  input_sha256=digest(document), protocol_sha256=digest(protocol),
                  branches={k: v.used for k, v in budgets.items()},
                  learning_usage=[r["learning_usage"] for r in event_rows],
                  transfer_usage=[r["usage"] for r in transfers],
                  original_task_scores="retained_unmodified_in_events",
                  wall_seconds=elapsed_time() - wall_started,
                  replay_wall_seconds=sum(replay_costs), replay_attempts=len(replay_costs),
                  transfer_replay_attempts=sum(int(t.get(k) is not None) for t in transfers for k in ("old", "new")),
                  model_calls=sum(u["calls"] for u in all_usage),
                  model_input_tokens=sum(u["input_tokens"] for u in all_usage),
                  model_output_tokens=sum(u["output_tokens"] for u in all_usage),
                  cache_hits=0, replay_cache_enabled=False,
                  regression_scope="previously_accepted_natural_witness_prefixes",
                  prior_conditional_state_behavior_regression="not_exhaustively_replayed_requires_semantic_review",
                  gpu_peak_memory="not_measured_by_local_orchestrator",
                  gpu_busy_seconds="not_measured", gpu_reserved_seconds="not_measured",
                  cpu_process_seconds="not_measured", disk_bytes_before_final_report=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
                  human_edits=0, audit_minutes=sum(r["audit_minutes"] for r in event_rows))
    atomic_json(output / "coverage_and_cost.json", report)
    (output / "P1_report.md").write_text(
        "# P1 mechanism study\n\n" + f"Decision: **{report['decision']}**.\n\n"
        + f"A witnesses: {report['witness_states']} states / {report['witness_scenarios']} scenarios. "
          f"B mechanisms: {report['mechanisms']}. C independent improvements: {report['transfer_improvements']}.\n\n"
        + "Every attempted candidate, rejection, unknown and budget failure is retained. "
          "This is an exploratory mechanism decision, not a native test result or a statistical claim. "
          "P2 is never started automatically.\n", encoding="utf-8")
    log.info(f"Study complete | A {'PASS' if report['A'] else 'NOT MET'} | "
             f"B {'PASS' if report['B'] else 'NOT MET'} | C {'PASS' if report['C'] else 'NOT MET'}")
    log.info(f"Cost: {len(replay_costs)} replay attempts | {sum(replay_costs):.2f}s replay time | "
             f"{report['wall_seconds']:.2f}s total wall time")
    log.info("Reports saved. P2 has not been started.")
    log.close()
    return report


def summarize(witnesses, probes, updates, transfers, events):
    real = [w for w in witnesses if w.get("evidence_origin") == "AppWorld"
            and w.get("source") in NATURAL and w.get("recovery_success") is True]
    # Several successful candidates at one state do not become several witnesses.
    states = {(w["task_id"], w["event_id"]) for w in real}
    scenarios = {w["scenario_id"] for w in real}
    mechanisms = {p["mechanism_signature"] for p in probes if p.get("valid") is True
                  and p.get("compensation_supported") is True and p.get("mechanism_signature")
                  and p.get("conditional_effect") is True and p.get("evidence_origin") == "AppWorld"}
    real_updates = [u for u in updates if u["validation"].get("accepted") is True
                    and u["validation"].get("evidence_origin") == "AppWorld"]
    improved = [t for t in transfers if t.get("independent") is True and t.get("source") == "AppWorld"
                and t.get("improved") is True and t.get("behavior_changed") is True and t.get("knowledge_changed") is True]
    regressions = [t for t in transfers if t.get("old") and t.get("new")
                   and t["old"].get("success") is True and t["new"].get("success") is not True]
    substantive = [u for u in real_updates if u.get("is_revision") is True]
    a, b, c = (len(states) >= 3 and len(scenarios) >= 2, len(mechanisms) >= 2,
               len(real_updates) >= 2 and bool(substantive) and bool(improved) and not regressions)
    origins = {w.get("evidence_origin", "unknown") for w in witnesses}
    origin = next(iter(origins)) if len(origins) == 1 else ("mixed" if origins else "unknown")
    return {"schema_version": 1, "stage": "P1", "evidence_origin": origin,
            "A": a, "B": b, "C": c, "decision": "eligible_for_P2_review" if a and b and c else "stop_or_analyze_existing_evidence",
            "witness_states": len(states), "witness_scenarios": len(scenarios), "mechanisms": len(mechanisms),
            "committed_updates": len(real_updates), "transfer_improvements": len(improved),
            "candidate_attempts": len(witnesses), "events": len(events),
            "mechanism_count_basis": "normalized_program_and_condition_structure_requires_semantic_review",
            "unresolved_events": sum(e["status"] == "unresolved" for e in events),
            "native_test_run": False, "p2_started": False}
