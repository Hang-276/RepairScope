# English run logs

All operator-facing run and experiment-process messages use concise English.
Experiments run on the server; the local checkout is used to implement and
verify code. Offline software fixtures must remain explicitly synthetic.

Each P0/P1 run writes an append-only UTF-8 `run.log` in its output directory.
The same status line is printed to standard error and flushed immediately.
Each line includes a UTC timestamp, elapsed time, stage, and English severity.
UTC timestamps are independent of the operator's local time zone so logs can
be compared across servers. The default CLI prints a compact English final
summary; `--json` keeps standard output available for a structured report.
Logs use plain text without ANSI colors. Machine-readable JSON/JSONL evidence
remains separate and retains its original content and identities.

Illustrative formatting only; these lines are not experiment results:

```text
<UTC timestamp> | +<elapsed>s | [P0] INFO | Reading existing asset manifests.
<UTC timestamp> | +<elapsed>s | [P0] WARNING | Required replay evidence is unavailable.
<UTC timestamp> | +<elapsed>s | [P1] INFO | Candidate checks: <current>/<total> (<percent>%); receipt recorded
<UTC timestamp> | +<elapsed>s | [P1] ERROR | Execution stopped; inspect protected receipts.
```

The per-run `RunLog(output_dir, stage, quiet=False)` API provides `info`,
`warning`, `error`, `progress(label, current, total, detail='')`, and `close`.
It also supports a context manager. `quiet=True` suppresses console status
while preserving the file. Opening an existing log appends; it does not
overwrite history or change the run's independent input/output safeguards.
No global logging handlers are installed, so separate runs do not duplicate
each other's messages.

These conventions apply to runtime status and experiment-process labels.
They do not restrict the language used for research discussion with the user.

Callers must pass short English status text, counts, and general reasons.
Do not pass raw prompts, model responses, repair programs, checker contents,
task instructions, entity/answer literals, credentials, or evidence objects.
The logger does not automatically identify or redact protected text. Inspect
the private machine artifacts when detail is needed, and keep `runs/` out of
the public repository. Newlines and terminal control characters in status
messages are normalized to keep one event per line.
