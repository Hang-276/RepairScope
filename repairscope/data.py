"""JSONL evidence IO, grouped isolation and contiguous rollout windows."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import validate_transition


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_transition(record)
            except (ValueError, TypeError) as error:
                raise ValueError(f'{path}:{line_number}: {error}') from error
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            for record in records:
                validate_transition(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_dataset(records: Iterable[Mapping[str, Any]], allow_synthetic: bool = False) -> dict[str, Any]:
    records = list(records)
    source_splits, group_splits, pair_groups = {}, {}, {}
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        validate_transition(record)
        if record['evidence_origin'] == 'synthetic' and not allow_synthetic:
            raise ValueError('Synthetic fixture data is forbidden in scientific training/evaluation unless allow_synthetic is explicit')
        split, group, task = record['split'], record['group_id'], record['task_id']
        for table, identity, label in ((source_splits, task, 'source task'), (group_splits, group, 'group')):
            if identity in table and table[identity] != split:
                raise ValueError(f'{label} {identity} crosses splits')
            table[identity] = split
        source = record.get('source_task_id', task)
        if source in source_splits and source_splits[source] != split:
            raise ValueError(f'source task {source} crosses splits')
        source_splits[source] = split
        if record.get('source_split', split) != split:
            raise ValueError('A condition/branch may not change its official source split')
        pair = record.get('cf_pair_id')
        if pair:
            identity = task, group, split
            if pair in pair_groups and pair_groups[pair] != identity:
                raise ValueError(f'counterfactual siblings {pair} cross task/group/split')
            pair_groups[pair] = identity
            if split != 'train' and record.get('used_for_training', False):
                raise ValueError('dev/test branches cannot train the Predictor')
        counts[split] += 1
    return {'records': len(records), 'groups': len(group_splits), 'tasks': len(source_splits), 'counterfactual_pairs': len(pair_groups), 'splits': dict(counts), 'evidence_origins': sorted({r['evidence_origin'] for r in records})}


def make_windows(records: Iterable[Mapping[str, Any]], horizon: int) -> list[list[dict[str, Any]]]:
    if horizon not in (1, 2, 4):
        raise ValueError('Proposal rollout horizons are H=1,2,4')
    streams: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        validate_transition(record)
        # CF branches are separate streams. Never concatenate siblings/conditions.
        identity = tuple(record.get(k) for k in ('task_id', 'group_id', 'split', 'trajectory_id', 'cf_pair_id', 'cf_condition', 'cf_role'))
        streams[identity].append(record)
    windows = []
    for stream in streams.values():
        stream = sorted(stream, key=lambda r: r['step'])
        steps = [r['step'] for r in stream]
        if len(steps) != len(set(steps)):
            raise ValueError('Duplicate steps in a trajectory; supply distinct trajectory_id values')
        for start in range(len(stream) - horizon + 1):
            window = stream[start:start + horizon]
            if [r['step'] for r in window] != list(range(window[0]['step'], window[0]['step'] + horizon)):
                continue
            # Discontinuity means a reset, intervention or corrupted prefix.
            if any(a['next_state'] != b['state'] for a, b in zip(window, window[1:])):
                continue
            windows.append(deepcopy(window))
    return windows


def grouped_partition(records: Iterable[Mapping[str, Any]], dev_fraction: float = 0.1, seed: int = 0) -> dict[str, list[dict[str, Any]]]:
    """Optional train-internal holdout; official dev/test membership is preserved.

    This does not relabel official train records as official dev. Instead returns
    `train` and `train_holdout` buckets while keeping each record's source split.
    """
    if not 0 <= dev_fraction < 1:
        raise ValueError('dev_fraction must lie in [0,1)')
    records = list(records)
    validate_dataset(records, allow_synthetic=True)
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # Union both task and declared group identities, so variants sharing either
    # cannot land in opposite internal partitions.
    parent: dict[str, str] = {}
    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]
    for record in records:
        task = 'task:' + record.get('source_task_id', record['task_id'])
        group = 'group:' + record['group_id']
        left, right = find(task), find(group)
        parent[max(left, right)] = min(left, right)
    for record in records:
        identity = find('task:' + record.get('source_task_id', record['task_id']))
        fraction = int(hashlib.sha256(f'{seed}:{identity}'.encode()).hexdigest()[:16], 16) / 2**64
        bucket = 'train_holdout' if record['split'] == 'train' and fraction < dev_fraction else record['split']
        result[bucket].append(deepcopy(record))
    return dict(result)
