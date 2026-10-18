"""Masked transition, free-rollout, boundary, escape and paired-effect losses."""
from __future__ import annotations

from collections import defaultdict

import torch
from torch.nn import functional as F

from .tensorizer import canonical, known
from .schema import semantic_diff


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1)


def transition_losses(output: dict, batch: dict, config: dict) -> dict:
    weights = config.get("loss", {})
    mask = batch["audit_mask"] & batch["mask"]
    # Coverage remains an auxiliary prior. The v2 decision objective supplies
    # additional, direct budget-competitive gradients to this same selector.
    scope = F.binary_cross_entropy_with_logits(output["scope_logits"], batch["target_changed"], reduction="none", pos_weight=torch.tensor(float(weights.get("scope_positive_weight", 2.0)), device=mask.device))
    scope_loss = masked_mean(scope, mask)
    # Every auditable field supervises the inexpensive heads, while the applied
    # prediction remains hard-budgeted.  This supplies a signal for omitted fields.
    op_loss = masked_mean(F.cross_entropy(output["op_logits"].transpose(1, 2), batch["target_op"], reduction="none"), mask)
    numeric_mask = mask & (batch["kind"] == 0) & (batch["target_exists"] > 0) & (batch["target_null"] < 0.5)
    number_loss = masked_mean(F.smooth_l1_loss(output["numeric"], batch["target_numeric"], reduction="none"), numeric_mask)
    copy_mask = mask & (batch["kind"] != 0) & (batch["target_value"] >= 0) & (batch["target_exists"] > 0) & (batch["target_null"] < 0.5)
    copy_loss = masked_mean(F.cross_entropy(output["value_logits"].transpose(1, 2), batch["target_value"], reduction="none", ignore_index=-100), copy_mask)
    goal_valid = batch["goal_mask"] & (batch["target_goal"] >= 0)
    goal_loss = masked_mean(F.cross_entropy(output["goal_logits"].transpose(1, 2), batch["target_goal"], reduction="none", ignore_index=-100), goal_valid)
    recorded_escape = masked_mean(F.binary_cross_entropy_with_logits(output["recorded_escape_logits"], batch["target_escape"], reduction="none"), batch["escape_mask"])
    # Reselected scopes receive newly derived labels from the saved full diff.
    # Partial audits without an observed escape never supply a negative label.
    new_targets, new_valid = [], []
    for i, record in enumerate(batch["records"]):
        selected = {key for j, key in enumerate(batch["metadata"][i]["keys"]) if bool(output["scope"][i, j])}
        audit = record.get("audit", {})
        escaped = bool(set(audit.get("changes", [])) - selected)
        new_targets.append(float(escaped))
        new_valid.append(escaped or bool(audit.get("complete", False)))
    target = torch.tensor(new_targets, dtype=output["escape_logits"].dtype, device=mask.device)
    valid = torch.tensor(new_valid, dtype=torch.bool, device=mask.device)
    selected_escape = masked_mean(F.binary_cross_entropy_with_logits(output["escape_logits"], target, reduction="none"), valid)
    expansion = masked_mean(F.binary_cross_entropy(output["expansion"].clamp(1e-6, 1-1e-6), target, reduction="none"), valid)
    escape_loss = (recorded_escape + selected_escape + expansion) / 3
    null_loss = masked_mean(F.binary_cross_entropy_with_logits(output["null_logits"], batch["target_null"], reduction="none"), mask & batch["nullable"] & (batch["target_exists"] > 0))
    patch_loss = op_loss + number_loss + copy_loss + null_loss
    step = float(weights.get("scope", 1)) * scope_loss + float(weights.get("patch", 1)) * patch_loss + float(weights.get("goal", 1)) * goal_loss
    goal_escape, goal_escape_count, goal_escape_positive = goal_escape_loss(output, batch, config) if float(weights.get("goal_relevant_escape", 0)) > 0 else (output["scope_logits"].sum() * 0, 0, 0)
    return {"step": step, "scope": scope_loss, "patch": patch_loss, "goal": goal_loss, "escape": escape_loss,
            "goal_relevant_escape": goal_escape, "goal_escape_valid": goal_escape_count, "goal_escape_positive": goal_escape_positive,
            "op": op_loss, "number": number_loss, "copy": copy_loss}


def goal_escape_loss(output, batch, config):
    """Reliable goal damage outside scope; an unknown remainder masks negatives."""
    from .decision import goal_relevant_evidence
    losses, count, positives = [], 0, 0
    for i, record in enumerate(batch["records"]):
        evidence = goal_relevant_evidence(record, config)
        keys = set(evidence.get("observed_positive_keys", evidence.get("keys", [])))
        for scope, logit in ((set(record.get("scope", [])), output["recorded_goal_escape_logits"][i]),
                             ({key for j, key in enumerate(batch["metadata"][i]["keys"]) if bool(output["scope"][i, j])}, output["goal_escape_logits"][i])):
            escaped = bool(keys - scope)
            if escaped or evidence["valid"]:
                count += 1
                positives += int(escaped)
                losses.append(F.binary_cross_entropy_with_logits(logit, logit.new_tensor(float(escaped))))
        selected = {key for j, key in enumerate(batch["metadata"][i]["keys"]) if bool(output["scope"][i, j])}
        escaped = bool(keys - selected)
        if escaped or evidence["valid"]:
            p = output["goal_expansion"][i].clamp(1e-6, 1-1e-6)
            losses.append(F.binary_cross_entropy(p, p.new_tensor(float(escaped))))
    zero = output["scope_logits"].sum() * 0
    return torch.stack(losses).mean() if losses else zero, count, positives


def rollout_state_loss(output: dict, batch: dict) -> torch.Tensor:
    state, mask = output["soft_state"], batch["audit_mask"] & batch["mask"]
    exists = masked_mean(F.binary_cross_entropy(state["exists"].clamp(1e-6, 1-1e-6), batch["target_exists"], reduction="none"), mask)
    number = masked_mean(F.smooth_l1_loss(state["numeric"], batch["target_numeric"], reduction="none"), mask & (batch["kind"] == 0) & (batch["target_exists"] > 0) & (batch["target_null"] < 0.5))
    value_mask = mask & (batch["kind"] != 0) & (batch["target_value"] >= 0) & (batch["target_exists"] > 0) & (batch["target_null"] < 0.5)
    safe_target = batch["target_value"].clamp_min(0)
    probability = state["values"].gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    value = masked_mean(-probability.clamp_min(1e-8).log(), value_mask)
    null = masked_mean(F.binary_cross_entropy(state["null"].clamp(1e-6, 1-1e-6), batch["target_null"], reduction="none"), mask & batch["nullable"] & (batch["target_exists"] > 0))
    return exists + number + value + null


def _aligned_distribution(output, batch, row, field, vocabulary):
    table = batch["metadata"][row]["tables"][field]
    # Constant alignment matrices route gradients through copied soft values.
    route = torch.zeros(output["soft_state"]["values"].shape[-1], len(vocabulary), device=output["scope_logits"].device)
    for i, value in enumerate(table):
        route[i, vocabulary[canonical(value)]] = 1.0
    return output["soft_state"]["values"][row, field] @ route


def _support(output, batch, left, right, key, config):
    metas = batch["metadata"]
    li, ri = metas[left]["keys"].index(key), metas[right]["keys"].index(key)
    field = metas[left]["fields"][li]
    ls, rs = output["soft_state"], output["soft_state"]
    existence = (ls["exists"][left, li] - rs["exists"][right, ri]).abs()
    if field.get("kind") == "number":
        threshold = max(float(field.get("semantic_threshold", config.get("loss", {}).get("numeric_support_threshold", 0.1))), 1e-6)
        value = 1 - torch.exp(-(ls["numeric"][left, li] - rs["numeric"][right, ri]).abs() / threshold)
    else:
        tokens = [canonical(v) for v in metas[left]["tables"][li] + metas[right]["tables"][ri]]
        vocabulary = {token: i for i, token in enumerate(dict.fromkeys(tokens))}
        lp = _aligned_distribution(output, batch, left, li, vocabulary)
        rp = _aligned_distribution(output, batch, right, ri, vocabulary)
        value = 0.5 * (lp - rp).abs().sum()
    null_difference = (ls["null"][left, li] - rs["null"][right, ri]).abs()
    both_nonnull = (1-ls["null"][left, li]) * (1-rs["null"][right, ri])
    return (1 - (1-existence) * (1-null_difference) * (1-value.clamp(0, 1) * both_nonnull)).clamp(1e-6, 1-1e-6)


def _target_difference(batch, left, right, key):
    records = batch["records"]
    if not records[left].get("action_valid", True) or not records[right].get("action_valid", True):
        return None
    a = {x["key"]: x for x in (records[left].get("audit_next_state") or records[left].get("next_state", {})).get("fields", [])}.get(key)
    b = {x["key"]: x for x in (records[right].get("audit_next_state") or records[right].get("next_state", {})).get("fields", [])}.get(key)
    def audited(record):
        audit = record.get("audit", {})
        return key in audit.get("field_domain", []) or key in audit.get("changes", []) or (audit.get("complete", False) and not audit.get("field_domain"))
    if not known(a) or not known(b) or not audited(records[left]) or not audited(records[right]):
        return None
    return float(bool(semantic_diff({"fields": [a]}, {"fields": [b]})))


def _matched_prefix(records: list[dict], left: int, right: int) -> bool:
    a, b = records[left], records[right]
    if any(a.get(key) != b.get(key) for key in ("task_id", "group_id", "split", "source_task_id", "source_split")):
        return False
    if a.get("state") != b.get("state") or a.get("history", []) != b.get("history", []):
        return False
    # If hidden offline pre-state evidence exists on both branches, compare the
    # fields both branches actually stored. Compact audit states intentionally
    # omit unchanged private fields, so a field stored by one sibling and not
    # the other is not evidence of a different prefix; overlapping fields must
    # still agree.
    if a.get("audit_state") is not None and b.get("audit_state") is not None:
        left = {field["key"]: field for field in a["audit_state"].get("fields", [])}
        right = {field["key"]: field for field in b["audit_state"].get("fields", [])}
        for key in set(left) & set(right):
            if left[key] != right[key]:
                return False
    for key in ("prefix_id", "checkpoint_id", "source_prefix_id"):
        if key in a or key in b:
            if a.get(key) != b.get(key):
                return False
    if a.get("prefix_verified") is False or b.get("prefix_verified") is False:
        return False
    return True


def _same_action(left: dict, right: dict) -> bool:
    keys = ("name", "arguments", "code")
    return {key: left.get(key) for key in keys} == {key: right.get(key) for key in keys}


def counterfactual_losses(output: dict, batch: dict, config: dict) -> dict:
    groups = defaultdict(dict)
    for i, record in enumerate(batch["records"]):
        pair = record.get("cf_pair_id")
        condition, role = record.get("cf_condition"), record.get("cf_role")
        if pair and condition in ("c0", "c1") and role in ("action", "reference"):
            groups[pair][(condition, role)] = i
    boundary, supports, effects = [], [], []
    for group in groups.values():
        condition_scores = {}
        for condition in ("c0", "c1"):
            if (condition, "action") not in group or (condition, "reference") not in group:
                continue
            left, right = group[(condition, "action")], group[(condition, "reference")]
            if not _matched_prefix(batch["records"], left, right):
                continue
            common = set(batch["metadata"][left]["keys"]) & set(batch["metadata"][right]["keys"])
            condition_scores[condition] = {}
            for key in sorted(common):
                target = _target_difference(batch, left, right, key)
                if target is None:
                    continue
                support = _support(output, batch, left, right, key, config)
                supports.append(F.binary_cross_entropy(support, support.new_tensor(target)))
                condition_scores[condition][key] = (support, target)
                li = batch["metadata"][left]["keys"].index(key)
                ri = batch["metadata"][right]["keys"].index(key)
                if batch["metadata"][left]["fields"][li].get("kind") == "number" and batch["target_null"][left, li] < .5 and batch["target_null"][right, ri] < .5:
                    predicted_effect = output["soft_state"]["numeric"][left, li] - output["soft_state"]["numeric"][right, ri]
                    target_effect = batch["target_numeric"][left, li] - batch["target_numeric"][right, ri]
                    effects.append(F.smooth_l1_loss(predicted_effect, target_effect))
                elif batch["target_value"][left, li] >= 0 and batch["target_value"][right, ri] >= 0:
                    tables = batch["metadata"][left]["tables"][li] + batch["metadata"][right]["tables"][ri]
                    vocabulary = {token: i for i, token in enumerate(dict.fromkeys(canonical(v) for v in tables))}
                    lp = _aligned_distribution(output, batch, left, li, vocabulary)
                    rp = _aligned_distribution(output, batch, right, ri, vocabulary)
                    target_l, target_r = torch.zeros_like(lp), torch.zeros_like(rp)
                    token_l = canonical(batch["metadata"][left]["tables"][li][int(batch["target_value"][left, li])])
                    token_r = canonical(batch["metadata"][right]["tables"][ri][int(batch["target_value"][right, ri])])
                    target_l[vocabulary[token_l]], target_r[vocabulary[token_r]] = 1, 1
                    effects.append(F.mse_loss(lp-rp, target_l-target_r))
        compatible_actions = all(_same_action(batch["records"][group[("c0", role)]]["action"], batch["records"][group[("c1", role)]]["action"]) for role in ("action", "reference")) if "c0" in condition_scores and "c1" in condition_scores else False
        if "c0" in condition_scores and "c1" in condition_scores and compatible_actions:
            for key in sorted(condition_scores["c0"].keys() & condition_scores["c1"].keys()):
                p0, y0 = condition_scores["c0"][key]
                p1, y1 = condition_scores["c1"][key]
                boundary.append(((p1-p0) - (y1-y0)) ** 2)
    zero = output["scope_logits"].sum() * 0
    mean = lambda values: torch.stack(values).mean() if values else zero
    return {"boundary": mean(boundary), "paired": mean(supports) + mean(effects),
            "boundary_fields": len(boundary), "paired_fields": len(supports)}
