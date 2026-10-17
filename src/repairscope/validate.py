"""Fixed, fail-closed validation of an automatically proposed local rule update."""

from __future__ import annotations

from datetime import datetime, timezone

from .knowledge import _copy, _digest, validate_rule


def validate_update(old_rule: dict | None, new_rule: dict, evidence: list[dict], context: dict) -> dict:
    """Return a bound acceptance receipt; this performs no environment execution.

    Supporting evidence needs ``executed/valid/recovery_success=True`` and
    ``known_harm=False``. Regression checks need ``valid/passed=True`` and
    ``known_harm=False``. Unknown or absent values cannot certify an update.
    An accepted synthetic-fixture receipt remains explicitly synthetic.
    """
    reasons: list[str] = []
    if not isinstance(context, dict):
        context = {}
        reasons.append("invalid_context")
    if not isinstance(evidence, list) or any(not isinstance(item, dict) for item in evidence):
        evidence = []
        reasons.append("invalid_evidence_list")
    valid_schema = True
    try:
        validate_rule(new_rule, context.get("forbidden_literals", ()))
        if old_rule is not None:
            validate_rule(old_rule)
    except (TypeError, ValueError, KeyError) as exc:
        valid_schema = False
        reasons.append(f"invalid_rule: {exc}")
    if context.get("task_scored") is not True:
        reasons.append("original_task_not_scored")
    if context.get("automatic") is not True:
        reasons.append("update_not_automatic")
    frozen_before, frozen_after = context.get("frozen_before"), context.get("frozen_after")
    if not isinstance(frozen_before, str) or not frozen_before or frozen_before != frozen_after:
        reasons.append("frozen_protocol_mutated_or_unknown")
    for key in ("human_task_edits", "human_rule_edits", "human_intervention_edits"):
        if type(context.get(key)) is not int or context[key] != 0:
            reasons.append(f"nonzero_or_unknown_{key}")
    checks = context.get("regression_checks")
    if not isinstance(checks, list) or any(not isinstance(check, dict) for check in checks):
        checks = []
        reasons.append("invalid_regression_checks")
    if old_rule is not None and not checks:
        reasons.append("missing_regression_checks")
    for check in checks:
        if check.get("valid") is not True or check.get("passed") is not True or check.get("known_harm") is not False:
            reasons.append("failed_or_unknown_regression_check")
            break
    by_id: dict[str, dict] = {}
    sources: set[str] = set()
    for item in evidence:
        evidence_id = item.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id or evidence_id in by_id:
            reasons.append("invalid_or_duplicate_evidence_id")
            continue
        by_id[evidence_id] = item
        source = item.get("source")
        if source not in {"AppWorld", "synthetic_fixture"}:
            reasons.append("unsupported_or_unknown_evidence_source")
        else:
            sources.add(source)
        if source == "AppWorld" and item.get("split") != "train":
            reasons.append("nontraining_or_unknown_appworld_partition")
        if item.get("split") in {"dev", "test", "test_normal", "test_challenge"}:
            reasons.append("evaluation_evidence_cannot_update_knowledge")
    if valid_schema:
        if old_rule is not None:
            if new_rule["rule_id"] != old_rule["rule_id"] or new_rule["version"] != old_rule["version"] + 1:
                reasons.append("invalid_rule_lineage")
            for key in ("evidence_ids", "counterexample_ids"):
                if not set(old_rule[key]) <= set(new_rule[key]):
                    reasons.append(f"deleted_{key}")
            semantic_keys = {"intent", "applicability", "repair", "goal_effects", "status"}
            if all(new_rule[key] == old_rule[key] for key in semantic_keys):
                reasons.append("no_substantive_rule_change")
        elif new_rule["version"] != 1:
            reasons.append("new_rule_must_start_at_version_one")
        old_ids = set() if old_rule is None else set(old_rule["evidence_ids"] + old_rule["counterexample_ids"])
        referenced = set(new_rule["evidence_ids"] + new_rule["counterexample_ids"])
        if not referenced <= old_ids | set(by_id):
            reasons.append("missing_referenced_evidence")
        support = [by_id[eid] for eid in new_rule["evidence_ids"] if eid in by_id]
        qualifies = lambda item: (
            item.get("executed") is True and item.get("valid") is True
            and item.get("recovery_success") is True and item.get("known_harm") is False
            and item.get("source") in {"AppWorld", "synthetic_fixture"}
        )
        if not any(qualifies(item) for item in support):
            reasons.append("no_executed_valid_supporting_improvement")
        if any(not qualifies(item) for item in support):
            reasons.append("invalid_or_harmful_supporting_evidence")
    origin = next(iter(sources)) if len(sources) == 1 else ("mixed" if sources else "unknown")
    def safe_digest(value):
        try:
            return _digest(value)
        except (TypeError, ValueError):
            return None
    diff = {}
    if valid_schema:
        for key in new_rule:
            previous = None if old_rule is None else old_rule.get(key)
            if previous != new_rule[key]:
                diff[key] = {"before": previous, "after": new_rule[key]}
    receipt = {
        "accepted": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "checked_rule_hash": safe_digest(new_rule),
        "checked_parent_hash": safe_digest(old_rule),
        "rule_id": new_rule.get("rule_id") if isinstance(new_rule, dict) else None,
        "old_version": old_rule.get("version") if isinstance(old_rule, dict) else None,
        "new_version": new_rule.get("version") if isinstance(new_rule, dict) else None,
        "diff": _copy(diff),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "automatic": context.get("automatic") is True,
        "task_scored": context.get("task_scored") is True,
        "frozen_before": frozen_before,
        "frozen_after": frozen_after,
        "evidence_ids": sorted(by_id),
        "evidence_sources": sorted(sources),
        "evidence_origin": origin,
        "research_evidence": sources == {"AppWorld"},
        "supporting_and_opposing_evidence": _copy(evidence),
        "regression_checks": _copy(checks),
        "human_task_edits": context.get("human_task_edits"),
        "human_rule_edits": context.get("human_rule_edits"),
        "human_intervention_edits": context.get("human_intervention_edits"),
    }
    return receipt
