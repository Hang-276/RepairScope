"""Explicit local audit and P1 entry points. Imports perform no experiments."""
from __future__ import annotations

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="repairscope", description="RepairScope-Evo: P0/P1 mechanism study")
    parser.add_argument("--version", action="version", version="RepairScope-Evo 0.2.0")
    commands = parser.add_subparsers(dest="command", required=True)
    p0 = commands.add_parser("p0", help="Audit existing assets; no new tasks or model calls")
    p0.add_argument("--assets", required=True)
    p0.add_argument("--output-dir", required=True)
    p1 = commands.add_parser("p1", help="Execute the frozen bounded train-only mechanism study")
    p1.add_argument("--p0-dir", required=True)
    p1.add_argument("--events", required=True)
    p1.add_argument("--output-dir", required=True)
    p1.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    smoke = commands.add_parser("fixture-smoke", help="Artificial CPU checks, never real P1 evidence")
    smoke.add_argument("--output-dir", required=True)
    inspect = commands.add_parser("inspect", help="Verify frozen P0 hashes without opening task contents")
    inspect.add_argument("--p0-dir", required=True)
    for command in (p0, p1, smoke, inspect):
        command.add_argument("--json", action="store_true", help="Write the full machine-readable receipt to stdout")
    args = parser.parse_args(argv)
    try:
        if args.command == "p0":
            from .p0 import audit_assets
            result = audit_assets(args.assets, args.output_dir)
            exit_code = 0 if result["status"] == "passed" else 2
        elif args.command == "p1":
            from .p1 import run_p1
            result = run_p1(args.p0_dir, args.events, args.output_dir, args.base_url)
            exit_code = 0  # A scientifically negative, complete run is still a valid report.
        elif args.command == "fixture-smoke":
            from .fixture import smoke
            result = smoke(args.output_dir)
            exit_code = 0 if result["status"] == "passed" else 1
        else:
            from .p0 import load_p0
            assets, protocol, manifest, _ = load_p0(args.p0_dir)
            result = {"status": assets["status"], "protocol_sha256": assets["protocol_sha256"],
                      "train_tasks": len(manifest["tasks"]), "stage": protocol["stage"], "p2_enabled": False}
            exit_code = 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"Blocked: {error}\n")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "p1":
        print(f"P1 complete | A: {'PASS' if result['A'] else 'NOT MET'} | "
              f"B: {'PASS' if result['B'] else 'NOT MET'} | C: {'PASS' if result['C'] else 'NOT MET'}")
        print(f"Replay attempts: {result['replay_attempts']} | Model calls: {result['model_calls']} | "
              f"Output tokens: {result['model_output_tokens']:,} | Wall time: {result['wall_seconds']:.2f}s")
        print(f"Reports: {args.output_dir}/P1_report.md and coverage_and_cost.json")
    elif args.command == "fixture-smoke":
        print(f"Software smoke: {result['status'].upper()} | Evidence: SYNTHETIC FIXTURE")
        print(f"Scientific validation: NO | Receipt: {args.output_dir}/receipt.json")
    else:
        print(f"P0 {'audit' if args.command == 'p0' else 'frozen assets'}: {result['status'].upper()}")
        if args.command == "p0":
            print(f"Unresolved checks: {len(result['reasons'])} | Report: {args.output_dir}/P0_report.md")
        else:
            print(f"Train tasks: {result['train_tasks']} | Stage: {result['stage']}")
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
