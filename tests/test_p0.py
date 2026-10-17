"""Exercise the real asset-audit path using a temporary Git/data fixture.

These documents intentionally imitate receipt contracts. They do not contain
AppWorld tasks, initialize AppWorld, or establish real P0/P1 research readiness.
"""

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from repairscope.p0 import audit_assets, load_p0
from repairscope.protocol import APPWORLD_COMMIT, default_protocol, digest, file_digest


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git(repository, *arguments):
    return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True,
                                   stderr=subprocess.PIPE).strip()


@pytest.fixture
def asset_fixture(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init", "-q")
    (repository / "retained.txt").write_text("existing work\n", encoding="utf-8")
    git(repository, "add", "retained.txt")
    git(repository, "update-index", "--add", "--cacheinfo", "160000", APPWORLD_COMMIT, "vendor/appworld")
    git(repository, "-c", "user.name=P0 Test Fixture", "-c", "user.email=p0-fixture@example.invalid",
        "commit", "-qm", "synthetic asset identity fixture")
    environment = {"appworld_commit": APPWORLD_COMMIT, "appworld_version": "0.2.0",
                   "test_fixture_only": True}
    model = {"name": "Qwen/Qwen3-8B-AWQ", "revision": "a" * 40, "test_fixture_only": True}
    tasks = [{"task_id": f"fixture_task_{i}", "scenario_id": f"fixture_scenario_{i}", "split": "train",
              "source": "natural_failure", "task_sha256": digest(f"synthetic task {i}"),
              "trajectory_sha256": digest(f"synthetic trajectory {i}")} for i in range(2)]
    goal_tasks = {}
    for task in tasks:
        goal_tasks[task["task_id"]] = {
            "split": "train", "instruction": "Update alpha and preserve beta.",
            "goals": [{"id": "fix", "kind": "state_write", "evidence_span": "Update alpha"},
                      {"id": "keep", "kind": "preservation_constraint", "evidence_span": "preserve beta"}]}
    documents = {
        "model_lock": model, "environment_manifest": environment,
        "train_manifest": {"tasks": tasks},
        "goals": {"extractor_frozen": True, "source": "existing_D1", "tasks": goal_tasks},
        "replay_receipt": {"ok": True, "evidence_origin": "AppWorld", "source": "natural_failure",
                           "split": "train", "prefix_verified": True, "task_id": tasks[0]["task_id"],
                           "scenario_id": tasks[0]["scenario_id"], "test_fixture_only": True,
                           "identity": {"environment_sha256": digest(environment),
                                        "task_sha256": tasks[0]["task_sha256"],
                                        "goals_sha256": digest(goal_tasks[tasks[0]["task_id"]]),
                                        "mapping_sha256": digest("synthetic checker mapping"),
                                        "protocol_sha256": digest("previous audited execution protocol")}},
        "service_receipt": {"accepted": True, "context_length": 32000, "c2_truncation_verified": True,
                            "c5_verified": True, "model_revision": model["revision"], "test_fixture_only": True},
    }
    spec = {"repository": "repository"}
    for key, value in documents.items():
        path = tmp_path / "input_assets" / f"{key}.json"
        write_json(path, value)
        spec[key] = {"path": str(path.relative_to(tmp_path)), "sha256": file_digest(path)}
    spec_path = tmp_path / "existing_assets.json"
    write_json(spec_path, spec)
    return {"root": tmp_path, "repository": repository, "spec": spec, "spec_path": spec_path,
            "documents": documents, "output": tmp_path / "audit_output"}


def replace_document(fixture, key, document):
    path = fixture["root"] / fixture["spec"][key]["path"]
    write_json(path, document)
    fixture["spec"][key]["sha256"] = file_digest(path)
    write_json(fixture["spec_path"], fixture["spec"])


def test_passing_audit_records_actual_git_identity_and_freezes_inputs(asset_fixture):
    fixture = asset_fixture
    result = audit_assets(fixture["spec_path"], fixture["output"])
    assert result["status"] == "passed", result["reasons"]
    assert result["repository"]["head"] == git(fixture["repository"], "rev-parse", "HEAD")
    assert result["repository"]["appworld_commit"] == APPWORLD_COMMIT
    assets, protocol, manifest, goals = load_p0(fixture["output"])
    assert assets == result
    assert protocol["execution"]["test_enabled"] is False and protocol["execution"]["p2_enabled"] is False
    assert protocol["model"]["revision"] == "a" * 40
    assert goals == fixture["documents"]["goals"]
    assert manifest["task_order"] == ["fixture_task_0", "fixture_task_1"]
    assert (fixture["output"] / "protocol_migration.md").read_text().count("superseded_before_completion") == 1
    report = (fixture["output"] / "P0_report.md").read_text()
    assert "no new task executed" in report and "P1 has not run" in report


def test_audit_preserves_dirty_work_instead_of_resetting_it(asset_fixture):
    fixture = asset_fixture
    tracked = fixture["repository"] / "retained.txt"
    tracked.write_text("uncommitted user work\n", encoding="utf-8")
    result = audit_assets(fixture["spec_path"], fixture["output"])
    assert result["status"] == "passed"
    assert "retained.txt" in result["repository"]["dirty_status"]
    assert result["repository"]["dirty_diff_sha256"] == digest(git(fixture["repository"], "diff", "HEAD"))
    assert tracked.read_text() == "uncommitted user work\n"


def test_missing_assets_produce_blocked_report_without_fake_pass(tmp_path):
    spec = tmp_path / "missing_assets.json"
    write_json(spec, {"repository": "missing_repository"})
    output = tmp_path / "blocked"
    result = audit_assets(spec, output)
    assert result["status"] == "blocked" and result["reasons"]
    assert all(value["verified"] is False for value in result["verification"].values())
    assert "blocked" in (output / "P0_report.md").read_text()
    with pytest.raises(ValueError, match="P0 is blocked"):
        load_p0(output)


def test_input_hash_tamper_is_blocked_even_when_content_looks_valid(asset_fixture):
    fixture = asset_fixture
    path = fixture["root"] / fixture["spec"]["model_lock"]["path"]
    changed = deepcopy(fixture["documents"]["model_lock"])
    changed["revision"] = "b" * 40
    write_json(path, changed)
    result = audit_assets(fixture["spec_path"], fixture["output"])
    assert result["status"] == "blocked"
    assert result["verification"]["model_lock"]["verified"] is False
    assert any("model_lock file hash mismatch" in reason for reason in result["reasons"])


@pytest.mark.parametrize("partition", ["dev", "test_normal", "test_challenge"])
def test_nontrain_manifest_cannot_become_learning_allowlist(asset_fixture, partition):
    fixture = asset_fixture
    manifest = deepcopy(fixture["documents"]["train_manifest"])
    manifest["tasks"][0]["split"] = partition
    replace_document(fixture, "train_manifest", manifest)
    result = audit_assets(fixture["spec_path"], fixture["output"])
    assert result["status"] == "blocked"
    assert any("train boundary" in reason for reason in result["reasons"])
    with pytest.raises(ValueError, match="blocked"):
        load_p0(fixture["output"])


@pytest.mark.parametrize("mutation", ["unfrozen", "wrong_source", "duplicate_id", "empty_span", "invented_span", "nonpersistent", "extra_dev"])
def test_goal_freeze_rejects_invalid_or_contaminated_targets(asset_fixture, mutation):
    fixture = asset_fixture
    goals = deepcopy(fixture["documents"]["goals"])
    item = goals["tasks"]["fixture_task_0"]
    if mutation == "unfrozen":
        goals["extractor_frozen"] = False
    elif mutation == "wrong_source":
        goals["source"] = "fresh_reprompt_until_eligible"
    elif mutation == "duplicate_id":
        item["goals"][1]["id"] = item["goals"][0]["id"]
    elif mutation == "empty_span":
        item["goals"][0]["evidence_span"] = ""
    elif mutation == "invented_span":
        item["goals"][0]["evidence_span"] = "unrequested hidden protection"
    elif mutation == "nonpersistent":
        item["goals"][1]["kind"] = "final_answer"
    else:
        goals["tasks"]["extra_dev_task"] = {"split": "dev", "instruction": "not learning data", "goals": []}
    replace_document(fixture, "goals", goals)
    result = audit_assets(fixture["spec_path"], fixture["output"])
    assert result["status"] == "blocked"
    assert any("goal freezing" in reason for reason in result["reasons"])


@pytest.mark.parametrize("filename", ["protocol.yaml", "data_manifest.json", "goals.lock.json"])
def test_frozen_output_tamper_is_refused_at_load_time(asset_fixture, filename):
    fixture = asset_fixture
    assert audit_assets(fixture["spec_path"], fixture["output"])["status"] == "passed"
    path = fixture["output"] / filename
    value = json.loads(path.read_text())
    value["unauthorized_after_freeze"] = True
    write_json(path, value)
    with pytest.raises(ValueError, match="Frozen P0"):
        load_p0(fixture["output"])


def test_rehashed_protocol_change_still_cannot_enable_test_or_expand_budget(asset_fixture):
    fixture = asset_fixture
    assert audit_assets(fixture["spec_path"], fixture["output"])["status"] == "passed"
    protocol_path = fixture["output"] / "protocol.yaml"
    protocol = json.loads(protocol_path.read_text())
    protocol["execution"]["test_enabled"] = True
    protocol["p1"]["branches_per_event"] = 1200
    write_json(protocol_path, protocol)
    assets_path = fixture["output"] / "assets.json"
    assets = json.loads(assets_path.read_text())
    assets["protocol_sha256"] = digest(protocol)
    write_json(assets_path, assets)
    with pytest.raises(ValueError, match="protocol drift"):
        load_p0(fixture["output"])


def test_existing_audit_receipts_cannot_be_overwritten(asset_fixture):
    fixture = asset_fixture
    audit_assets(fixture["spec_path"], fixture["output"])
    original = (fixture["output"] / "assets.json").read_bytes()
    with pytest.raises(ValueError, match="cannot be overwritten"):
        audit_assets(fixture["spec_path"], fixture["output"])
    assert (fixture["output"] / "assets.json").read_bytes() == original


@pytest.mark.parametrize("mutation", ["split", "source", "task_hash", "goal_hash", "environment_hash", "prefix"])
def test_replay_receipt_is_bound_to_real_train_asset_identity(asset_fixture, mutation):
    fixture = asset_fixture
    replay = deepcopy(fixture["documents"]["replay_receipt"])
    if mutation == "split":
        replay["split"] = "dev"
    elif mutation == "source":
        replay["source"] = "synthetic_fixture"
    elif mutation == "prefix":
        replay["prefix_verified"] = False
    else:
        key = {"task_hash": "task_sha256", "goal_hash": "goals_sha256",
               "environment_hash": "environment_sha256"}[mutation]
        replay["identity"][key] = "0" * 64
    replace_document(fixture, "replay_receipt", replay)
    result = audit_assets(fixture["spec_path"], fixture["output"])
    assert result["status"] == "blocked"
    assert any("train prefix replay" in reason for reason in result["reasons"])


def test_changed_staged_upstream_cannot_hide_behind_pinned_head(asset_fixture):
    fixture = asset_fixture
    git(fixture["repository"], "update-index", "--add", "--cacheinfo", "160000", "b" * 40, "vendor/appworld")
    result = audit_assets(fixture["spec_path"], fixture["output"])
    assert result["status"] == "blocked"
    assert any("index gitlink" in reason for reason in result["reasons"])


def test_initialized_upstream_worktree_must_match_pinned_revision(asset_fixture):
    fixture = asset_fixture
    upstream = fixture["repository"] / "vendor/appworld"
    upstream.mkdir(parents=True)
    git(upstream, "init", "-q")
    (upstream / "fixture.txt").write_text("no AppWorld code\n", encoding="utf-8")
    git(upstream, "add", "fixture.txt")
    git(upstream, "-c", "user.name=P0 Test Fixture", "-c", "user.email=p0-fixture@example.invalid",
        "commit", "-qm", "different synthetic upstream revision")
    result = audit_assets(fixture["spec_path"], fixture["output"])
    assert result["status"] == "blocked"
    assert any("Initialized AppWorld" in reason for reason in result["reasons"])


def test_hashed_nonobject_document_is_reported_blocked(asset_fixture):
    fixture = asset_fixture
    replace_document(fixture, "model_lock", [])
    result = audit_assets(fixture["spec_path"], fixture["output"])
    assert result["status"] == "blocked"
    assert any("model_lock must be a JSON object" in reason for reason in result["reasons"])


def test_committed_configuration_exactly_matches_frozen_runtime_defaults():
    path = Path(__file__).resolve().parents[1] / "configs/evo_v1.yaml"
    assert json.loads(path.read_text(encoding="utf-8")) == default_protocol()
