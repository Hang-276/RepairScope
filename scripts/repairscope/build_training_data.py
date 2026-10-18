"""Combine the one-time natural and CF receipts without altering source splits."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repairscope.data import read_jsonl, write_jsonl, validate_dataset

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-synthetic', action='store_true')
    args = parser.parse_args()
    records = [r for source in args.inputs for r in read_jsonl(source)]
    if any(r['split'] != 'train' for r in records):
        parser.error('Only train records belong in the shared training set')
    report = validate_dataset(records, allow_synthetic=args.allow_synthetic)
    write_jsonl(args.output, records)
    print(report)

if __name__ == '__main__':
    main()
