"""Small per-run English log with no global handlers or implicit evidence dumps."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time


class RunLog:
    """Append readable status lines to ``run.log`` and optionally stderr.

    Callers supply short English status messages, never raw model prompts,
    programs, task text, checker details, credentials, or evidence payloads.
    Machine-readable evidence belongs in its own protected artifacts.
    """

    def __init__(self, output_dir, stage, quiet=False):
        if not isinstance(stage, str) or not re.fullmatch(r"[A-Z][A-Z0-9_-]*", stage):
            raise ValueError("Log stage must use uppercase ASCII letters, digits, '_' or '-'")
        self.stage = stage
        self.quiet = bool(quiet)
        self._started = time.monotonic()
        self._closed = False
        self.path = Path(output_dir) / "run.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def _write(self, level, message):
        if self._closed:
            raise ValueError("Run log is closed")
        if not isinstance(message, str):
            raise TypeError("Log message must be English status text")
        # Keep one event per line and strip terminal control characters. The
        # caller remains responsible for excluding protected evidence content.
        message = " ".join(message.split())
        message = "".join(char for char in message if char.isprintable())
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        elapsed = max(0.0, time.monotonic() - self._started)
        line = f"{stamp} | +{elapsed:.2f}s | [{self.stage}] {level} | {message}\n"
        self._file.write(line)
        self._file.flush()
        if not self.quiet:
            sys.stderr.write(line)
            sys.stderr.flush()

    def info(self, message):
        self._write("INFO", message)

    def warning(self, message):
        self._write("WARNING", message)

    def error(self, message):
        self._write("ERROR", message)

    def progress(self, label, current, total, detail=""):
        if not isinstance(label, str) or not isinstance(detail, str):
            raise TypeError("Progress label and detail must be English status text")
        if type(current) is not int or type(total) is not int or not 0 <= current <= total or total <= 0:
            raise ValueError("Progress requires integer counts with 0 <= current <= total and total > 0")
        percent = current / total * 100
        message = f"{label}: {current}/{total} ({percent:.0f}%)"
        if detail:
            message += f"; {detail}"
        self.info(message)

    def close(self):
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
