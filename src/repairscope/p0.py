"""Audit supplied existing assets without collecting data or opening test tasks."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess

from .io import atomic_json
from .logging import RunLog
from .protocol import (APPWORLD_COMMIT, SOURCES, default_protocol, digest,
                       file_digest, read_json, require_train, round_robin)


def _checked_document(spec, key, base):
    entry = spec[key]
    path = (base / entry["path"]).resolve()
    if file_digest(path) != entry["sha256"]:
        raise ValueError(f"{key} file hash mismatch")
    document = read_json(path)
    if not isinstance(document, dict):
        raise ValueError(f"{key} must be a JSON object")
    return document


def audit_assets(asset_file, output_dir):
    asset_file = Path(asset_file).resolve()
    spec, base = read_json(asset_file), asset_file.parent
    output = Path(output_dir)
    if output.exists():
        raise ValueError("P0 output directory already exists; receipts cannot be overwritten")
    output.mkdir(parents=True)
    log = RunLog(output, "P0")
    log.info("Asset audit started. Existing files only; no model calls or task execution.")
    protocol = default_protocol()
    reasons = []
    assets = {"schema_version": 1, "observed_at": datetime.now(timezone.utc).isoformat(),
              "input_sha256": file_digest(asset_file), "verification": {},
              "source": "operator_supplied_existing_assets"}
    documents = {}
    for number, key in enumerate(("model_lock", "environment_manifest", "train_manifest", "goals", "replay_receipt", "service_receipt"), 1):
        try:
            documents[key] = _checked_document(spec, key, base)
            assets["verification"][key] = {"sha256": spec[key]["sha256"], "verified": True}
        except (KeyError, OSError, ValueError, TypeError) as error:
            reasons.append(f"{key}: {error}")
            assets["verification"][key] = {"verified": False}
        log.progress("Asset checks", number, 6, detail=f"{key}: {'OK' if assets['verification'][key]['verified'] else 'MISSING OR INVALID'}")
    # Reading git identity never resets, pulls, installs or starts server services.
    try:
        repo = (base / spec["repository"]).resolve()
        def git(*args):
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
        assets["repository"] = {"head": git("rev-parse", "HEAD"),
                                "dirty_status": git("status", "--porcelain"),
                                "dirty_diff_sha256": digest(git("diff", "HEAD")),
                                "appworld_commit": git("rev-parse", "HEAD:vendor/appworld")}
        index_entry = git("ls-files", "--stage", "vendor/appworld").split()
        if len(index_entry) < 2 or index_entry[1] != APPWORLD_COMMIT:
            reasons.append("AppWorld index gitlink differs from frozen upstream")
        if (repo / "vendor/appworld/.git").exists():
            current_upstream = git("-C", str(repo / "vendor/appworld"), "rev-parse", "HEAD")
            assets["repository"]["appworld_worktree_commit"] = current_upstream
            if current_upstream != APPWORLD_COMMIT:
                reasons.append("Initialized AppWorld checkout differs from frozen upstream")
        if assets["repository"]["appworld_commit"] != APPWORLD_COMMIT:
            reasons.append("AppWorld submodule identity differs from the frozen proposal")
    except (KeyError, OSError, subprocess.CalledProcessError) as error:
        reasons.append(f"repository identity unavailable: {type(error).__name__}")
    tasks = documents.get("train_manifest", {}).get("tasks", [])
    try:
        if not tasks:
            raise ValueError("nonempty existing train task manifest required")
        if len({t["task_id"] for t in tasks}) != len(tasks):
            raise ValueError("duplicate train task IDs")
        for task in tasks:
            require_train(task["split"])
            if not task.get("scenario_id") or task.get("source") not in SOURCES:
                raise ValueError("task scenario/source identity missing")
            if task["source"] == "synthetic_fixture":
                raise ValueError("fixture cannot pass real P0")
            for key in ("task_sha256", "trajectory_sha256"):
                if not re.fullmatch("[0-9a-f]{64}", task.get(key, "")):
                    raise ValueError(f"{key} requires a full SHA256")
        order = round_robin(tasks)
    except (KeyError, ValueError, TypeError) as error:
        reasons.append(f"train boundary: {error}")
        order = []
    model = documents.get("model_lock", {})
    if model.get("name") != protocol["model"]["name"] or not re.fullmatch("[0-9a-f]{40}", model.get("revision", "")):
        reasons.append("full immutable Qwen3-8B-AWQ model revision is not locked")
    environment = documents.get("environment_manifest", {})
    if environment.get("appworld_commit") != APPWORLD_COMMIT or environment.get("appworld_version") != "0.2.0":
        reasons.append("environment does not confirm pinned AppWorld 0.2.0")
    service = documents.get("service_receipt", {})
    if not (service.get("accepted") is True and service.get("context_length") == 32000
            and service.get("c2_truncation_verified") is True
            and service.get("c5_verified") is True and service.get("model_revision") == model.get("revision")):
        reasons.append("actual C2/C5 service receipt has not confirmed the frozen configuration")
    goals = documents.get("goals", {})
    try:
        if goals.get("extractor_frozen") is not True or goals.get("source") != "existing_D1":
            raise ValueError("goals must reuse frozen existing D1 extraction")
        entries = goals["tasks"]
        for task_id in order:
            item = entries[task_id]
            require_train(item["split"])
            if (not item["goals"] or len({g["id"] for g in item["goals"]}) != len(item["goals"])
                    or any(not g.get("id") or not g.get("evidence_span")
                           or g["evidence_span"] not in item["instruction"] for g in item["goals"])):
                raise ValueError("goals require exact public instruction evidence")
            if sum(g.get("kind") in {"state_write", "preservation_constraint"} for g in item["goals"]) < 2:
                raise ValueError("Evo eligibility requires two independent persistent goals/constraints")
        if set(entries) != set(order):
            raise ValueError("frozen goals must contain exactly the train allowlist, no dev/test")
    except (KeyError, ValueError, TypeError) as error:
        reasons.append(f"goal freezing: {error}")
    replay = documents.get("replay_receipt", {})
    replay_task = next((t for t in tasks if t.get("task_id") == replay.get("task_id")), {})
    replay_goals = goals.get("tasks", {}).get(replay.get("task_id"), {})
    if not (replay.get("ok") is True and replay.get("evidence_origin") == "AppWorld"
            and replay.get("prefix_verified") is True and replay.get("task_id") in order
            and replay.get("split") == "train" and replay.get("source") in {"natural_failure", "natural_regression", "natural_incomplete"}
            and replay.get("identity", {}).get("task_sha256") == replay_task.get("task_sha256")
            and replay.get("identity", {}).get("goals_sha256") == digest(replay_goals)
            and replay.get("identity", {}).get("environment_sha256") == digest(environment)):
        reasons.append("no matching trustworthy existing train prefix replay receipt")
    protocol["task_order"] = order
    protocol["model"]["revision"] = model.get("revision")
    protocol["asset_hashes"] = {key: value["sha256"] for key, value in assets["verification"].items() if value.get("verified")}
    manifest = {"schema_version": 1, "learning_split": "train", "tasks": tasks,
                "task_order": order, "dev_policy": "development_only_no_persistent_learning",
                "test_policy": "sealed_metadata_only_until_P4"}
    assets.update(status="passed" if not reasons else "blocked", reasons=reasons,
                  protocol_sha256=digest(protocol), data_manifest_sha256=digest(manifest),
                  goals_sha256=digest(goals), environment_sha256=digest(environment))
    atomic_json(output / "assets.json", assets)
    atomic_json(output / "protocol.yaml", protocol)  # JSON is a strict YAML subset.
    atomic_json(output / "data_manifest.json", manifest)
    atomic_json(output / "goals.lock.json", goals)
    (output / "protocol_migration.md").write_text(
        "# Protocol migration\n\nOld GATE 2: `superseded_before_completion`.\n"
        "The previous predictor method is retired. Historical results remain in Git history.\n"
        "No environment or service was reset. Existing C2/C5 receipts remain authoritative.\n", encoding="utf-8")
    (output / "P0_report.md").write_text(
        f"# P0 asset audit\n\nStatus: **{assets['status']}**\n\n"
        "Inputs: existing operator-supplied asset files; no new task executed.\n\n"
        f"Train tasks indexed: {len(order)}. Protocol hash: `{assets['protocol_sha256']}`.\n\n"
        + ("Unresolved:\n\n" + "\n".join(f"- {r}" for r in reasons) if reasons else
           "Existing identities and supplied replay evidence satisfy the P0 checks. P1 has not run.")
        + "\n", encoding="utf-8")
    log.info(f"Train tasks indexed: {len(order)}. Protocol and goal locks saved.")
    if reasons:
        log.warning(f"Audit BLOCKED: {len(reasons)} unresolved checks. See P0_report.md for details.")
    else:
        log.info("Audit PASSED. P1 has not been executed.")
    log.close()
    return assets


def load_p0(directory):
    directory = Path(directory)
    assets = read_json(directory / "assets.json")
    protocol = read_json(directory / "protocol.yaml")
    manifest = read_json(directory / "data_manifest.json")
    goals = read_json(directory / "goals.lock.json")
    if assets.get("status") != "passed":
        raise ValueError("P0 is blocked: existing real assets must pass before P1")
    for key, value in (("protocol_sha256", protocol), ("data_manifest_sha256", manifest), ("goals_sha256", goals)):
        if digest(value) != assets[key]:
            raise ValueError(f"Frozen P0 {key} mismatch")
    # A modified limit or phase flag cannot silently enable later experiments.
    for key, value in default_protocol().items():
        actual = protocol[key]
        if key == "model":
            actual = {k: actual[k] for k in value}
        if actual != value:
            raise ValueError(f"P0 protocol drift: {key}")
    return assets, protocol, manifest, goals
