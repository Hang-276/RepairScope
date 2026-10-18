"""Summarize complete per-seed main results without mixing variants or fixtures."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics


def summarize(root: Path):
    rows = []
    for path in sorted(root.glob('*/seed_*/evaluation/summary.json')):
        value = json.loads(path.read_text())
        metrics = value.get('metrics', {})
        rows.append({'variant': path.parents[2].name, 'seed': path.parents[1].name,
            'TGC': value.get('TGC'), 'SGC': value.get('SGC'),
            'goal_regression_rate': metrics.get('goal_regression_rate'),
            'repair_new_damage_rate': metrics.get('repair_new_damage_rate'),
            'cost': value.get('cost'), 'complete_native_coverage': value.get('complete_native_coverage', False),
            'evidence_origins': ','.join(value.get('evidence_origins', [])), 'source': str(path)})
    variants = {}
    for row in rows:
        variants.setdefault(row['variant'], []).append(row)
    summary = {}
    for variant, seed_rows in variants.items():
        result = {'seeds': [r['seed'] for r in seed_rows], 'metrics': {}}
        for key in ('TGC', 'SGC', 'goal_regression_rate', 'repair_new_damage_rate', 'cost'):
            values = [r[key] for r in seed_rows if isinstance(r[key], (int, float))]
            result['metrics'][key] = {'mean': statistics.mean(values) if values else None,
                'std': statistics.stdev(values) if len(values) > 1 else None, 'n_seeds': len(values)}
        summary[variant] = result
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    rows, summary = summarize(args.run_root)
    if not rows:
        parser.error('No completed per-seed evaluation summaries found')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / 'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
