"""Cheap engineering gate before GPU spend, never evidence of publishability."""
import json
from pathlib import Path
from b2_appworld.datasets import collection_quality_report
from b2_appworld.io import atomic_json

reports = {}
errors = []
for split in ("train", "dev"):
    report = collection_quality_report(f"runs/labels/{split}.jsonl")
    reports[split] = report
    atomic_json(f"runs/labels/{split}.quality.json", report)
    if report["source_example_counts"].get("natural", 0) != report["num_examples"]:
        errors.append(f"{split}: main experiment data must be entirely natural")
    if report["fully_labeled_complete_candidate_pools"] < 10:
        errors.append(f"{split}: fewer than 10 fully labeled four-candidate pools")
    if report["states_with_01"] < 10 or report["states_with_10"] < 10:
        errors.append(f"{split}: fewer than 10 observed recovery or damage states")
    if report["states_with_differential_recovery_damage_counts"] < 10:
        errors.append(f"{split}: fewer than 10 states with decision-relevant candidate differences")
    if any(not entry["receipt_present"] or entry["receipt_evidence_origin"] != "AppWorld"
           for entry in report["source_files"]):
        errors.append(f"{split}: missing genuine AppWorld replay receipt")
    if split == "train" and report["splits"]["train"]["scenarios"] < 5:
        errors.append("train: fewer than five scenarios for held-out diagnostics")
audit_path = Path("runs/labels/mapping_audit.json")
if not audit_path.exists():
    errors.append("Missing documented semantic mapping audit; see docs/LABEL_AUDIT.md")
else:
    audit = json.loads(audit_path.read_text())
    # This receipt is a research annotation record, not a user approval flow.
    if audit.get("goals_reviewed", 0) < 30 or not audit.get("issues_resolved", False):
        errors.append("Mapping audit must document >=30 reviewed goals and resolved errors")
result = {"passed": not errors, "gate_kind": "engineering_only_not_statistical_validation", "errors": errors}
atomic_json("runs/labels/gate.json", result)
print(json.dumps(result, indent=2))
if errors:
    raise SystemExit(2)
