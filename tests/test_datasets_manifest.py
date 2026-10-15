import json

import pytest

from b2_appworld.datasets import EXPECTED_COUNTS, collection_quality_report, dataset_manifest
from b2_appworld.environment import AppWorldFactory
from b2_appworld.schemas import Candidate, Example, Goal
from b2_appworld.io import write_jsonl


def build_fixture(root):
    datasets = root / "data" / "datasets"
    datasets.mkdir(parents=True)
    offset = 0
    for split, count in EXPECTED_COUNTS.items():
        ids = [f"s{scenario:06}_{n}" for scenario in range(offset, offset + count // 3) for n in (1, 2, 3)]
        (datasets / f"{split}.txt").write_text("\n".join(x + ":tag" for x in ids))
        if split in {"train", "dev"}:
            for task_id in ids:
                task = root / "data" / "tasks" / task_id
                task.mkdir(parents=True)
                (task / "specs.json").write_text(json.dumps({"db_version": "0.2.0"}))
        offset += count // 3
    return datasets


def test_exact_counts_versions_and_no_test_task_content_required(tmp_path):
    build_fixture(tmp_path)
    manifest = dataset_manifest(tmp_path)
    assert manifest["test_contents_inspected"] is False
    assert manifest["splits"]["test_challenge"]["count"] == 417
    assert manifest["observed_train_dev_versions"] == ["0.2.0"]
    assert len(manifest["split_root_sha256"]) == 64


def test_cross_split_scenario_leakage_rejected(tmp_path):
    datasets = build_fixture(tmp_path)
    path = datasets / "dev.txt"
    text = path.read_text()
    path.write_text(text.replace("s000030", "s000000"))
    with pytest.raises(ValueError, match="leakage"):
        dataset_manifest(tmp_path)


def test_wrong_version_rejected(tmp_path):
    build_fixture(tmp_path)
    (tmp_path / "data/tasks/s000000_1/specs.json").write_text('{"db_version":"0.1.0"}')
    with pytest.raises(ValueError, match="0.2.0"):
        dataset_manifest(tmp_path)


def test_test_full_ground_truth_rejected_before_import(tmp_path):
    with pytest.raises(ValueError, match="train/dev"):
        with AppWorldFactory(tmp_path).create("x_1", "safe", "test_normal", full_ground_truth=True):
            pass


def quality_rows():
    goals = (Goal("x", "Set x", "Set x"), Goal("y", "keep y", "keep y"))
    rows = []
    for name, labels in zip(("original", "local", "composite", "broad"),
                            ((0, 3), (1, 3), (1, 2), (0, 2))):
        rows.append(Example("fake_1", "fake", "s1", "train", "Set x and keep y", [], goals,
                    Candidate(name, "pass", ("x",)), labels,
                    source="synthetic_fixture", before_fingerprint="same").to_dict())
    return rows


def test_quality_report_candidate_pools_and_transition_coverage(tmp_path):
    path = tmp_path / "examples.jsonl"
    write_jsonl(path, quality_rows())
    report = collection_quality_report(path)
    assert report["complete_candidate_pools"] == 1
    assert report["fully_labeled_complete_candidate_pools"] == 1
    assert report["states_with_01"] == report["states_with_10"] == 1
    assert report["states_with_both_01_and_10"] == 1
    assert report["unknown_label_fraction"] == 0
    assert report["class_counts"] == {"00": 2, "01": 2, "10": 2, "11": 2, "unknown": 0}
    assert any("synthetic" in warning for warning in report["warnings"])
    assert report["semantic_mapping_audit_required"] is True


def test_quality_report_rejects_inconsistent_before_labels(tmp_path):
    rows = quality_rows()
    rows[1]["labels"] = [3, 3]
    path = tmp_path / "examples.jsonl"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="before-goal"):
        collection_quality_report(path)


def test_quality_report_rejects_scenario_leakage(tmp_path):
    rows = quality_rows()
    other = dict(rows[0], task_id="fake_2", split="dev")
    rows.append(other)
    path = tmp_path / "examples.jsonl"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="Scenario leakage"):
        collection_quality_report(path)
