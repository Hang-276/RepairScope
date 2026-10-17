from __future__ import annotations

import re

import pytest

from repairscope.logging import RunLog


def test_status_is_flushed_to_file_and_console(tmp_path, capsys):
    log = RunLog(tmp_path / "run", "P1")
    log.info("Starting bounded branch checks.")
    saved = log.path.read_text(encoding="utf-8")
    console = capsys.readouterr()
    assert saved == console.err
    assert console.out == ""
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z \| \+\d+\.\d{2}s \| "
        r"\[P1\] INFO \| Starting bounded branch checks\.\n", saved
    )
    assert "\x1b" not in saved
    log.close()


def test_append_quiet_and_lifecycle(tmp_path, capsys):
    with RunLog(tmp_path, "P0", quiet=True) as first:
        first.warning("Required asset is unavailable.")
    with RunLog(tmp_path, "P1", quiet=True) as second:
        second.error("Execution stopped; inspect protected receipts.")
    lines = (tmp_path / "run.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "[P0] WARNING" in lines[0]
    assert "[P1] ERROR" in lines[1]
    console = capsys.readouterr()
    assert console.out == console.err == ""
    second.close()
    with pytest.raises(ValueError, match="closed"):
        second.info("This cannot be written.")


def test_progress_and_single_line_status(tmp_path, capsys):
    with RunLog(tmp_path, "P1") as log:
        log.progress("Branch checks", 1, 4, "status recorded")
        log.info("A status\nwith a line break.\tContinuing.")
        with pytest.raises(ValueError, match="integer counts"):
            log.progress("Branch checks", 5, 4)
        with pytest.raises(ValueError, match="integer counts"):
            log.progress("Branch checks", True, 4)
    saved = (tmp_path / "run.log").read_text(encoding="utf-8")
    console = capsys.readouterr()
    assert saved == console.err
    assert console.out == ""
    assert "Branch checks: 1/4 (25%); status recorded" in saved
    assert "A status with a line break. Continuing." in saved
    assert len(saved.splitlines()) == 2


def test_no_global_handlers_or_cross_run_duplicate_lines(tmp_path, capsys):
    with RunLog(tmp_path / "a", "P0") as first, RunLog(tmp_path / "b", "P1") as second:
        first.info("First run status.")
        second.info("Second run status.")
    console = capsys.readouterr()
    assert len(console.err.splitlines()) == 2
    assert console.out == ""
    assert "Second run" not in (tmp_path / "a/run.log").read_text()
    assert "First run" not in (tmp_path / "b/run.log").read_text()


def test_invalid_stage_does_not_create_output(tmp_path):
    output = tmp_path / "uncreated"
    with pytest.raises(ValueError, match="uppercase ASCII"):
        RunLog(output, "P1\nERROR")
    assert not output.exists()
