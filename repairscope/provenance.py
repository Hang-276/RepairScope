"""Receipts prevent fixture checks from being mistaken for live experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


UPSTREAM_URL = "https://github.com/kyle8581/WMA-Agents"
UPSTREAM_COMMIT = "dd894640204895b52f050c66cc0b58f08e00f9b8"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: str | Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(canonical_bytes(value) + b"\n")
    temp.replace(path)


def receipt(config: dict, records: list[dict] | None = None) -> dict:
    origins = sorted({r.get("evidence_origin", "unknown") for r in records or []})
    synthetic = config["environment"]["backend"] == "fixture" or any(
        "synthetic" in x or "fixture" in x for x in origins)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"schema_version": 1, "upstream_url": UPSTREAM_URL,
            "upstream_commit": UPSTREAM_COMMIT, "local_commit": commit,
            "working_tree_dirty": dirty, "config_sha256": digest(config),
            "records_sha256": digest(records) if records is not None else None,
            "record_count": len(records) if records is not None else None,
            "evidence_origin": "synthetic" if synthetic else "appworld",
            "is_live_result": False,
            "note": "Receipt alone is not a successful live environment measurement."}
