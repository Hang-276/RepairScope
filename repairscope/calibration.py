"""One dev-only threshold selection; test interaction never fits a threshold."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from .provenance import file_digest, write_json


def calibrate_threshold(predictions: str | Path, checkpoint: str | Path,
                        output: str | Path, max_check_rate: float = 0.25,
                        risk_source: str = "escape") -> dict:
    if not 0 <= max_check_rate <= 1:
        raise ValueError("max_check_rate must be in [0,1]")
    if risk_source not in {"escape", "goal_escape", "uncertainty"}:
        raise ValueError("risk_source must be escape, goal_escape or uncertainty")
    checkpoint_hash = file_digest(checkpoint)
    values = []
    with Path(predictions).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            # Use each observed dev transition once; longer windows are for the
            # main rollout metrics, not additional fitting observations.
            if row["horizon"] != 1 or row["offset"] != 1:
                continue
            if row["split"] != "dev":
                raise ValueError("Threshold fitting is restricted to official dev records")
            if row["checkpoint_id"] != checkpoint_hash:
                raise ValueError("Dev predictions and checkpoint have different hashes")
            prediction, audit = row["prediction"], row["label_receipt"]
            if risk_source == "goal_escape":
                label = audit.get("goal_escape_label")
                if label is None:
                    continue
                missed = bool(label)
            else:
                missed = bool(set(audit["changes"]) - set(prediction["scope"]))
                if not missed and not audit["complete"]:
                    continue  # A partial audit without a positive is unknown.
            probability = float(prediction[risk_source])
            if not math.isfinite(probability) or probability < 0:
                raise ValueError("Risk scores must be finite and nonnegative")
            values.append((probability, int(missed)))
    if not values:
        raise ValueError("No audited dev H=1 predictions available for threshold selection")
    candidates = sorted({score for score, _ in values} |
                        {math.nextafter(max(score for score, _ in values), math.inf)})
    positives = sum(label for _, label in values)
    feasible = []
    for threshold in candidates:
        selected = [(score, label) for score, label in values if score >= threshold]
        check_rate = len(selected) / len(values)
        if check_rate > max_check_rate + 1e-12:
            continue
        tp = sum(label for _, label in selected)
        recall = tp / positives if positives else 0.0
        precision = tp / len(selected) if selected else 0.0
        feasible.append((recall, precision, -check_rate, threshold))
    recall, precision, negative_rate, threshold = max(feasible)
    result = {"schema_version": 1, "split": "dev", "checkpoint_sha256": checkpoint_hash,
              "prediction_file_sha256": file_digest(predictions), "risk_source": risk_source,
              "risk_threshold": threshold, "max_check_rate": max_check_rate,
              "dev_records": len(values), "dev_escape_positives": positives,
              "dev_check_rate": -negative_rate,
              "dev_escape_recall": recall if positives else None,
              "dev_check_precision": precision, "fitted_once": True,
              "selection_rule": "Maximize escape recall under a fixed check-rate cap; ties prefer precision then fewer checks.",
              "probability_recalibration": False, "test_labels_used": False}
    output = Path(output)
    if output.exists():
        raise FileExistsError("Calibration output already exists; retain the frozen dev receipt")
    write_json(output, result)
    return result


def apply_calibration(config: dict, checkpoint: str | Path | None) -> dict:
    config = deepcopy(config)
    path = config.get("controller", {}).get("calibration_file")
    if not path:
        return config
    if checkpoint is None:
        raise ValueError("A frozen calibration receipt requires its checkpoint")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("split") != "dev" or value.get("test_labels_used") is not False:
        raise ValueError("Only a dev-only frozen threshold receipt is accepted")
    if value.get("checkpoint_sha256") != file_digest(checkpoint):
        raise ValueError("Calibration receipt does not belong to this checkpoint")
    expected_source = calibration_source(config)
    if value.get("risk_source") != expected_source:
        raise ValueError("Calibration risk source does not match the deployed controller")
    threshold = float(value["risk_threshold"])
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid calibrated threshold")
    config.setdefault("controller", {})["risk_threshold"] = threshold
    config["controller"]["calibration_receipt"] = value
    return config


def calibration_source(config):
    if float(config.get("controller", {}).get("escape_weight", 1)) <= 0:
        return "uncertainty"
    if config.get("model", {}).get("goal_sufficient", False) and float(config.get("loss", {}).get("goal_relevant_escape", 0)) > 0:
        return "goal_escape"
    return "escape"
