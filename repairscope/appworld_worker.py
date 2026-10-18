"""Server-only AppWorld integration, isolated from model-facing observations.

API verified against AppWorld revision 42b5bcf3cd334fee33f0c37c02070a9f5807add5:
AppWorld(task_id, experiment_name, load_ground_truth, random_seed), execute(code),
models.get_tracked_db_changes(), models[app].SQLModel.db.connection, save(), and
appworld.evaluator.evaluate_task(). Real execution is intentionally not imported
or invoked by the Mac smoke test.
"""
from __future__ import annotations

from copy import deepcopy
from fnmatch import fnmatchcase
import hashlib
import json
import multiprocessing as mp
import math
import os
from pathlib import Path
import re
import sqlite3
import time
import traceback
from typing import Any

from .environment import action_code, public_state_update, validate_condition
from .schema import evaluate_goals, field_map, semantic_diff, validate_state

_ERROR = re.compile(r'Traceback \(most recent call last\)|(?:^|\n)(?:ExecutionError|SyntaxError|Exception|ValueError|TypeError|HTTPException|NameError):|Execution failed', re.I)


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return {'__bytes__': value.hex()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Datetimes/UUIDs occur in public API metadata; string representation keeps
    # their public value without exposing arbitrary object attributes.
    return str(value)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _fingerprint(world: Any) -> str:
    # The audit change-log and its trigger inserts are instrumentation, not
    # business state. Exclude them so prefix fingerprints match across workers.
    changes = {}
    for app, statements in world.models.get_tracked_db_changes().items():
        changes[app] = [item for item in statements if '_rs_audit_changes' not in str(item[0])]
    return _digest({'task_id': world.task_id, 'changes': changes})


def _set_root(root: str) -> None:
    root = str(Path(root).expanduser().resolve())
    os.environ['APPWORLD_ROOT'] = root
    from appworld.common.path_store import path_store
    path_store.update_root(root)


def snapshot_world(world: Any, ignore: list[str] | None = None) -> dict[str, Any]:
    """Full SQLite audit, never a model observation or legal retrieval.

    Keys use app.table[primary-key].column. Null remains known null. Creation and
    deletion are aligned later with explicit tombstones. Only fixed declared
    filtering is accepted; timestamps are retained unless the run manifest says
    otherwise. Internal record_hash instrumentation is excluded by default.
    """
    ignore = ['*.record_hash', *(ignore or [])]
    fields = []
    for app, models in sorted(world.models.items()):
        connection = models.SQLModel.db.connection
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        for (table,) in tables:
            # SQLite FTS shadow tables are internal indexes, not semantic state.
            # Auditing them is both meaningless and prohibitively expensive on
            # the real AppWorld databases.
            if '_fts' in table or table == '_rs_audit_changes':
                continue
            quoted = '"' + table.replace('"', '""') + '"'
            columns = connection.execute('PRAGMA table_info(' + quoted + ')').fetchall()
            column_names = [column[1] for column in columns]
            primary = [column[1] for column in sorted(columns, key=lambda row: row[5]) if column[5]]
            cursor = connection.execute('SELECT rowid, * FROM ' + quoted) if not primary else connection.execute('SELECT * FROM ' + quoted)
            for raw_row in cursor.fetchall():
                row = dict(zip(column_names, raw_row if primary else raw_row[1:]))
                identity_values = [row[name] for name in primary] if primary else [raw_row[0]]
                identity = ','.join(str(v) for v in identity_values)
                entity = f'{app}.{table}[{identity}]'
                for column in columns:
                    name, sqltype = column[1], (column[2] or '').upper()
                    key = entity + '.' + name
                    if any(fnmatchcase(key, pattern) for pattern in ignore):
                        continue
                    value = row[name]
                    if name == 'id' or name.endswith('_id'):
                        kind = 'reference'
                        value = str(value) if value is not None else None
                    elif 'BOOL' in sqltype:
                        kind, value = 'bool', None if value is None else bool(value)
                    elif re.match(r'^(?:INT(?:EGER)?|BIGINT|SMALLINT|TINYINT|REAL|FLOAT|DOUBLE|NUMERIC|DECIMAL)\b', sqltype) or isinstance(value, (int, float)):
                        kind = 'number'
                    else:
                        kind = 'text'
                        value = value.hex() if isinstance(value, bytes) else str(value) if value is not None else None
                    field = {'key': key, 'entity': entity, 'kind': kind, 'known': True, 'exists': True, 'value': value, 'choices': []}
                    if value is None:
                        field['nullable'] = True
                    fields.append(field)
    state = {'fields': fields, 'relations': []}
    validate_state(state)
    return state


def align_audit_states(before: dict[str, Any], after: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    before, after = deepcopy(before), deepcopy(after)
    left, right = field_map(before), field_map(after)
    for key in sorted(set(left) | set(right)):
        if key not in left:
            left[key] = {**deepcopy(right[key]), 'value': None, 'exists': False, 'known': True}
        if key not in right:
            right[key] = {**deepcopy(left[key]), 'value': None, 'exists': False, 'known': True}
    before['fields'], after['fields'] = list(left.values()), list(right.values())
    return before, after


_MISSING = object()
_DELETED = object()
_AUDIT_LOG = '_rs_audit_changes'


def _table_schema(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    result = {}
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    for (table,) in tables:
        if '_fts' in table or table == _AUDIT_LOG:
            continue
        columns = connection.execute('PRAGMA table_info("' + table.replace('"', '""') + '")').fetchall()
        names = [row[1] for row in columns if row[1] != 'record_hash']
        if not names:
            continue
        result[table] = {
            'columns': names,
            'pk': [row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5]],
            'types': {row[1]: (row[2] or '').upper() for row in columns},
            'nullable': {row[1]: not bool(row[3]) for row in columns},
        }
    return result


def _install_change_tracking(world: Any) -> dict[str, Any]:
    """Create an in-memory baseline and record the tracked-statement cursor.

    AppWorld's own SQL tracker is used only to identify *which tables/rows may
    have been touched*. Net field changes are always obtained by comparing the
    actual before/after row values against the baseline and a sparse row cache.
    """
    tracking: dict[str, Any] = {'apps': {}, 'complete': True, 'rows': {},
                                'rowids': {}, 'descriptor': {'kind': 'appworld_initial_db'}}
    tracked = world.models.get_tracked_db_changes()
    for app in world.models.model_names():
        if app == 'SQLModel':
            continue
        connection = world.models[app].SQLModel.db.connection
        baseline = sqlite3.connect(':memory:')
        try:
            connection.commit()
        except sqlite3.Error:
            pass
        connection.backup(baseline)
        schema = _table_schema(connection)
        tracking['apps'][app] = {
            'connection': connection, 'baseline': baseline, 'schema': schema,
            'schema_version': connection.execute('PRAGMA schema_version').fetchone()[0],
        }
        tracking.setdefault('start', {})[app] = len(tracked.get(app, []))
    return tracking


def _remove_change_tracking(tracking: dict[str, Any] | None) -> None:
    if not tracking:
        return
    for info in tracking.get('apps', {}).values():
        baseline = info.get('baseline')
        if baseline is not None:
            try:
                baseline.close()
            except sqlite3.Error:
                pass


def _schema_versions(tracking: dict[str, Any]) -> dict[str, int]:
    return {app: info['connection'].execute('PRAGMA schema_version').fetchone()[0]
            for app, info in tracking.get('apps', {}).items()}


def _strip_identifier(value: str) -> str:
    return value.strip().strip('"').strip("'").split('.')[-1]


def _parse_conditions(where: str, parameters: tuple | list) -> list[tuple[str, Any]]:
    columns = re.findall(r'([A-Za-z_][A-Za-z0-9_."]*)\s*=\s*\?', where)
    columns = [_strip_identifier(column) for column in columns]
    if not columns or len(columns) > len(parameters):
        return []
    values = list(parameters)[-len(columns):]
    return list(zip(columns, values))


def _parse_statement(statement: str, parameters: Any, executemany: bool,
                     schema: dict[str, dict[str, Any]]) -> tuple[str | None, list[tuple[str, Any]], bool]:
    """Return (table, equalities, is_insert) for one tracked SQL statement."""
    text = ' '.join(str(statement).split())
    match = re.match(r'^INSERT(?: OR \w+)? INTO\s+("?[A-Za-z0-9_]+"?)\s*\(([^)]*)\)\s*VALUES', text, re.I)
    if match:
        table = _strip_identifier(match.group(1))
        columns = [_strip_identifier(column) for column in match.group(2).split(',')]
        rows = parameters if executemany and isinstance(parameters, list) else [parameters]
        equalities = []
        for row in rows:
            if not isinstance(row, (tuple, list)) or len(row) != len(columns):
                continue
            values = dict(zip(columns, row))
            equalities.extend((column, values[column]) for column in schema.get(table, {}).get('pk', [])
                              if column in values)
        return table, equalities, True
    match = re.match(r'^UPDATE\s+("?[A-Za-z0-9_]+"?)\s+SET\s+(.*?)\s+WHERE\s+(.*)$', text, re.I)
    if match:
        table = _strip_identifier(match.group(1))
        set_columns = [_strip_identifier(column) for column in
                       re.findall(r'([A-Za-z_][A-Za-z0-9_."]*)\s*=\s*\?', match.group(2))]
        if any(column in schema.get(table, {}).get('pk', []) for column in set_columns):
            return table, [], False
        rows = parameters if executemany and isinstance(parameters, list) else [parameters]
        equalities = []
        for row in rows:
            if not isinstance(row, (tuple, list)):
                continue
            pairs = _parse_conditions(match.group(3), row)
            equalities.extend(pairs)
        return table, equalities, False
    match = re.match(r'^DELETE\s+FROM\s+("?[A-Za-z0-9_]+"?)\s+WHERE\s+(.*)$', text, re.I)
    if match:
        table = _strip_identifier(match.group(1))
        rows = parameters if executemany and isinstance(parameters, list) else [parameters]
        equalities = []
        for row in rows:
            if isinstance(row, (tuple, list)):
                equalities.extend(_parse_conditions(match.group(2), row))
        return table, equalities, False
    return None, [], False


def _rowid_for_pk(tracking: dict[str, Any], app: str, table: str,
                  schema: dict[str, Any], pk: tuple[tuple[str, Any], ...],
                  connection: sqlite3.Connection | None = None) -> int | None:
    if not pk or [column for column, _ in pk] != schema['pk']:
        return None
    info = tracking['apps'][app]
    target = connection or info['connection']
    quoted = '"' + table.replace('"', '""') + '"'
    where = ' AND '.join('"' + column.replace('"', '""') + '"=?' for column, _ in pk)
    row = target.execute(f'SELECT rowid FROM {quoted} WHERE {where}',
                         [value for _, value in pk]).fetchone()
    return int(row[0]) if row is not None else None


def _query_row_by_pk(tracking: dict[str, Any], app: str, table: str,
                     schema: dict[str, Any], pk: tuple[tuple[str, Any], ...]
                     ) -> tuple[int | None, dict[str, Any] | None]:
    info = tracking['apps'][app]
    rowid = _rowid_for_pk(tracking, app, table, schema, pk)
    if rowid is None:
        rowid = _rowid_for_pk(tracking, app, table, schema, pk, info['baseline'])
    if rowid is None:
        return None, None
    return rowid, _query_row(info['connection'], table, rowid)


def _new_rowids(tracking: dict[str, Any], app: str, table: str) -> set[int]:
    key = (app, table)
    info = tracking['apps'][app]
    quoted = '"' + table.replace('"', '""') + '"'
    if key not in tracking['rowids']:
        tracking['rowids'][key] = {int(row[0]) for row in info['baseline'].execute(
            f'SELECT rowid FROM {quoted}')}
    live = {int(row[0]) for row in info['connection'].execute(f'SELECT rowid FROM {quoted}')}
    new = live - tracking['rowids'][key]
    tracking['rowids'][key] = live
    return new


def _collect_changed_rows(world: Any, tracking: dict[str, Any]
                          ) -> list[tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]]:
    tracked = world.models.get_tracked_db_changes()
    changed: list[tuple[str, str, dict | None, dict | None]] = []
    seen: set[tuple[str, str, int]] = set()
    for app, info in tracking['apps'].items():
        statements = tracked.get(app, [])
        start = tracking['start'].get(app, 0)
        new = statements[start:]
        tracking['start'][app] = len(statements)
        schema = info['schema']
        pk_changes: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
        insert_tables: set[str] = set()
        for statement, parameters, executemany in new:
            table, equalities, is_insert = _parse_statement(statement, parameters, executemany, schema)
            if table is None or table not in schema:
                tracking['complete'] = False
                continue
            if is_insert:
                insert_tables.add(table)
            if equalities:
                by_name = dict(equalities)
                if all(column in by_name for column in schema[table]['pk']):
                    pk_changes.add((table, tuple((column, by_name[column]) for column in schema[table]['pk'])))
                else:
                    tracking['complete'] = False
            elif not is_insert:
                tracking['complete'] = False
        for table, pk in pk_changes:
            rowid, after = _query_row_by_pk(tracking, app, table, schema[table], pk)
            before = tracking['rows'].get((app, table, rowid), _MISSING) if rowid is not None else _MISSING
            if before is _MISSING:
                before = (_query_row(info['baseline'], table, rowid) if rowid is not None else None)
            elif before is _DELETED:
                before = None
            if rowid is None:
                continue
            if (app, table, rowid) in seen:
                continue
            seen.add((app, table, rowid))
            changed.append((app, table, before, after))
        for table in insert_tables:
            for rowid in sorted(_new_rowids(tracking, app, table)):
                before = tracking['rows'].get((app, table, rowid), _MISSING)
                if before is _MISSING:
                    before = _query_row(info['baseline'], table, rowid)
                elif before is _DELETED:
                    before = None
                after = _query_row(info['connection'], table, rowid)
                if (app, table, rowid) in seen:
                    continue
                seen.add((app, table, rowid))
                changed.append((app, table, before, after))
    for app, table, before, after in changed:
        info = tracking['apps'][app]
        schema = info['schema'][table]
        identity = None
        if before is not None:
            identity = _rowid_from_row(before)
        elif after is not None:
            identity = _rowid_from_row(after)
        if identity is not None:
            tracking['rows'][(app, table, identity)] = after if after is not None else _DELETED
    return changed


def _rowid_from_row(row: dict[str, Any]) -> int | None:
    value = row.get('rowid')
    return int(value) if value is not None else None


def _query_row(connection: sqlite3.Connection, table: str, rowid: int) -> dict[str, Any] | None:
    quoted = '"' + table.replace('"', '""') + '"'
    cursor = connection.execute(f'SELECT rowid, * FROM {quoted} WHERE rowid=?', (int(rowid),))
    description = [column[0] for column in cursor.description]
    row = cursor.fetchone()
    return dict(zip(description, row)) if row is not None else None


def _row_fields(app: str, table: str, row: dict[str, Any], schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    identity_values = [row.get(name) for name in schema['pk']] if schema['pk'] else [row.get('rowid')]
    identity = ','.join(str(value) for value in identity_values)
    entity = f'{app}.{table}[{identity}]'
    fields = {}
    for column in schema['columns']:
        value = row.get(column)
        sqltype = schema['types'].get(column, '')
        if column == 'id' or column.endswith('_id'):
            kind = 'reference'
            value = str(value) if value is not None else None
        elif 'BOOL' in sqltype:
            kind = 'bool'
            value = None if value is None else bool(value)
        elif re.match(r'^(?:INT(?:EGER)?|BIGINT|SMALLINT|TINYINT|REAL|FLOAT|DOUBLE|NUMERIC|DECIMAL)\b', sqltype) \
                or (value is not None and isinstance(value, (int, float)) and not isinstance(value, bool)):
            kind = 'number'
        else:
            kind = 'text'
            if isinstance(value, bytes):
                value = value.hex()
            elif value is not None:
                value = str(value)
        field = {'key': entity + '.' + column, 'entity': entity, 'kind': kind, 'known': True,
                 'exists': True, 'value': value, 'choices': []}
        if value is None:
            field['nullable'] = True
        fields[field['key']] = field
    return fields


def _tombstone(field: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(field)
    out.update(known=True, exists=False, value=None)
    out.pop('choices', None)
    return out


def _changed_pairs(before_row: dict[str, Any] | None, after_row: dict[str, Any] | None,
                   app: str, table: str, schema: dict[str, Any]) -> dict[str, tuple[dict, dict]]:
    before_fields = _row_fields(app, table, before_row, schema) if before_row is not None else {}
    after_fields = _row_fields(app, table, after_row, schema) if after_row is not None else {}
    template = next(iter(after_fields.values()), None) or next(iter(before_fields.values()), None)
    pairs = {}
    for key in sorted(set(before_fields) | set(after_fields)):
        before = before_fields.get(key) or (deepcopy(template) if template else None)
        after = after_fields.get(key) or (deepcopy(template) if template else None)
        if before is None or after is None:
            continue
        if before_fields.get(key) is None:
            before = _tombstone(after)
        if after_fields.get(key) is None:
            after = _tombstone(before)
        if semantic_diff({'fields': [before]}, {'fields': [after]}):
            pairs[key] = (before, after)
    return pairs


def _parse_key(key: str) -> tuple[str, str, str, str] | None:
    match = re.match(r'^([^.]+)\.([^\[]+)\[([^\]]*)\]\.(.+)$', key)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3), match.group(4)


def _lookup_key_field(tracking: dict[str, Any], key: str,
                      template: dict[str, Any] | None = None) -> dict[str, Any] | None:
    parsed = _parse_key(key)
    if parsed is None:
        return None
    app, table, identity, _ = parsed
    info = tracking.get('apps', {}).get(app)
    if info is None or table not in info['schema']:
        return None
    schema = info['schema'][table]
    if not schema['pk']:
        return None
    values = identity.split(',')
    if len(values) != len(schema['pk']):
        return None
    quoted = '"' + table.replace('"', '""') + '"'
    where = ' AND '.join('"' + name.replace('"', '""') + '"=?' for name in schema['pk'])
    cursor = info['connection'].execute(f'SELECT rowid, * FROM {quoted} WHERE {where}', values)
    description = [column[0] for column in cursor.description]
    row = cursor.fetchone()
    if row is None:
        if template is None:
            return None
        return _tombstone(template)
    row = dict(zip(description, row))
    return _row_fields(app, table, row, schema).get(key)


def _compact_audit(world: Any, tracking: dict[str, Any], before_public: dict[str, Any],
                   after_public: dict[str, Any], before_versions: dict[str, int]) -> tuple[dict, dict, dict]:
    changed_rows = _collect_changed_rows(world, tracking)
    after_versions = _schema_versions(tracking)
    complete = bool(tracking.get('complete')) and before_versions == after_versions
    pairs: dict[str, tuple[dict, dict]] = {}
    for app, table, before_row, after_row in changed_rows:
        info = tracking.get('apps', {}).get(app)
        if info is None or table not in info['schema']:
            complete = False
            continue
        schema = info['schema'][table]
        pairs.update(_changed_pairs(before_row, after_row, app, table, schema))
    selected: dict[str, tuple[dict, dict]] = dict(pairs)
    public_fields = {}
    for state in (before_public, after_public):
        public_fields.update(field_map(state))
    for key, template in public_fields.items():
        if key in selected:
            continue
        field = _lookup_key_field(tracking, key, template)
        if field is not None:
            selected[key] = (field, field)
    before_fields = {key: deepcopy(before) for key, (before, _) in selected.items()}
    after_fields = {key: deepcopy(after) for key, (_, after) in selected.items()}
    audit = {
        'complete': bool(complete),
        'field_domain': sorted(selected),
        'changes': sorted(pairs),
        'model_version': 'unselected',
        'domain_kind': 'compact_sqlite' if complete else 'partial',
        'stored_scope': 'public+changed',
        'baseline_ref': deepcopy(tracking.get('descriptor', {})),
    }
    return audit, {'fields': list(before_fields.values()), 'relations': []}, {
        'fields': list(after_fields.values()), 'relations': []}


def _goal_state(world: Any, tracking: dict[str, Any], goals: list[dict[str, Any]],
                public_state: dict[str, Any], max_fields: int = 200000) -> dict[str, Any]:
    """Materialize only fields referenced by goals from the live public database."""
    from .schema import goal_patterns
    fields: dict[str, dict[str, Any]] = {}
    for goal in goals:
        patterns = goal_patterns(goal)
        if not patterns:
            kind = goal.get('kind', 'predicate')
            if kind == 'answer':
                patterns = ['supervisor.tasks[*].answer']
            elif kind == 'complete':
                patterns = ['supervisor.tasks[*].status']
        for pattern in patterns:
            parsed = re.match(r'^([^.]+)\.([^\[]+)\[([^\]]*)\]\.(.+)$', pattern)
            if parsed is None:
                continue
            app_pattern, table_pattern, identity_pattern, column_pattern = parsed.groups()
            for app, info in tracking.get('apps', {}).items():
                if not fnmatchcase(app, app_pattern):
                    continue
                for table, schema in info['schema'].items():
                    if not fnmatchcase(table, table_pattern):
                        continue
                    columns = [name for name in schema['columns'] if fnmatchcase(name, column_pattern)]
                    if not columns:
                        continue
                    quoted = '"' + table.replace('"', '""') + '"'
                    if identity_pattern == '*':
                        cursor = info['connection'].execute(f'SELECT rowid, * FROM {quoted}')
                    else:
                        if not schema['pk'] or len(identity_pattern.split(',')) != len(schema['pk']):
                            continue
                        where = ' AND '.join('"' + name.replace('"', '""') + '"=?' for name in schema['pk'])
                        cursor = info['connection'].execute(
                            f'SELECT rowid, * FROM {quoted} WHERE {where}', identity_pattern.split(','))
                    description = [column[0] for column in cursor.description]
                    for raw in cursor.fetchall():
                        row = dict(zip(description, raw))
                        row_fields = _row_fields(app, table, row, schema)
                        for column in columns:
                            key = f'{app}.{table}[{",".join(str(row.get(name)) for name in schema["pk"]) if schema["pk"] else row.get("rowid")}].{column}'
                            if key in row_fields:
                                fields[key] = row_fields[key]
                        if len(fields) > max_fields:
                            return {'fields': list(fields.values()), 'relations': []}
    # Public keys are legal observations and may be the only representation of
    # a goal in a state; merge them without overwriting audit fields.
    for key, field in field_map(public_state).items():
        fields.setdefault(key, deepcopy(field))
    return {'fields': list(fields.values()), 'relations': []}


def _goal_labels(world: Any, tracking: dict[str, Any] | None, goals: list[dict[str, Any]],
                 before_public: dict[str, Any], after_public: dict[str, Any]) -> dict[str, Any]:
    if not goals or tracking is None:
        return {}
    try:
        before_state = _goal_state(world, tracking, goals, before_public)
        after_state = _goal_state(world, tracking, goals, after_public)
        return {'goal_before': evaluate_goals(before_state, goals),
                'goal_after': evaluate_goals(after_state, goals, previous=before_state)}
    except (ValueError, KeyError, TypeError, sqlite3.Error) as error:
        return {'goal_error': f'{type(error).__name__}: {error}'}


def _task_override_digest(root: str, task_id: str) -> str | None:
    path = Path(root).expanduser() / 'data' / 'tasks' / task_id / 'dbs'
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob('*') if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(b'\0')
        digest.update(file.read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def _evaluate_entry(connection: Any, request: dict[str, Any]) -> None:
    try:
        _set_root(request['root'])
        from appworld.evaluator import evaluate_task
        tracker = evaluate_task(task_id=request['task_id'], experiment_name=request['experiment'], suppress_errors=False, save_report=True)
        data = _jsonable(tracker.to_dict())
        connection.send({'ok': True, 'result': {'task_id': request['task_id'], 'split': request['split'], 'evidence_origin': 'appworld',
                                               'tracker': data, 'TGC': float(data['success']), 'SGC': None,
                                               'requires_scenario_aggregation': True, 'evaluator_isolation': 'fresh_spawn'}})
    except BaseException as error:
        connection.send({'ok': False, 'error': f'{type(error).__name__}: {error}'})
    finally:
        connection.close()


def evaluate_fresh(world: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Disk evaluation in a fresh spawn avoids CachedDBHandler aliasing."""
    world.save()
    context = mp.get_context('spawn')
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_evaluate_entry, args=(child, {'root': request['config']['environment']['root'], 'task_id': request['task_id'], 'experiment': request['experiment'], 'split': request['split']}))
    process.start()
    child.close()
    try:
        timeout = request['config']['environment'].get('worker_timeout', 300)
        if not parent.poll(timeout):
            raise TimeoutError('Fresh evaluator timed out')
        response = parent.recv()
        if not response['ok']:
            raise RuntimeError(response['error'])
        return response['result']
    finally:
        parent.close()
        process.join(2)
        if process.is_alive():
            process.terminate()
            process.join(3)
        if process.is_alive():
            process.kill()
            process.join()


def _public_metadata(world: Any, request: dict[str, Any]) -> dict[str, Any]:
    task = world.task
    env = request['config']['environment']
    mapping = env.get('goal_map') or {}
    goals = mapping.get(request['task_id'], mapping.get('goals', [])) if isinstance(mapping, dict) else mapping
    if isinstance(goals, dict):
        goals = goals.get('goals', [])
    return {'instruction': task.instruction, 'supervisor': _jsonable(task.supervisor),
            'allowed_apps': list(task.allowed_apps), 'api_docs': _jsonable(dict(task.api_docs)),
            'goals': deepcopy(goals), 'read_only_actions': deepcopy(env.get('read_only_actions', [])),
            'evidence_origin': 'appworld', 'granularity': env.get('granularity', 'code_block'),
            'api_revision': env.get('appworld_revision'), 'audit_domain': 'SQLite fields, offline only'}


def _initial_state(metadata: dict[str, Any]) -> dict[str, Any]:
    # Only the same public instruction/supervisor available to the actor.
    from .environment import public_state_update
    state = {'fields': [{'key': 'task.instruction', 'kind': 'text', 'value': metadata['instruction'], 'known': True, 'exists': True, 'choices': []}], 'relations': []}
    return public_state_update(state, {'name': 'supervisor.public_identity', 'arguments': {}}, json.dumps(metadata['supervisor']))


def _execute(world: Any, action: dict[str, Any], state: dict[str, Any], env: dict[str, Any], offline: bool,
             budget_remaining: float | None = None, tracking: dict[str, Any] | None = None,
             goals: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if budget_remaining is not None and budget_remaining < 1:
        raise ValueError('Insufficient budget for one code block; charged_cost=0')
    tracker = world.requester.request_tracker
    previous_limit = tracker.max_num_requests
    if budget_remaining is not None:
        api_limit = max(0, math.floor(budget_remaining - 1))
        tracker.max_num_requests = min(previous_limit, api_limit) if previous_limit is not None else api_limit
    compact = bool(offline and tracking is not None and env.get('audit_mode', 'compact') == 'compact')
    before_versions = _schema_versions(tracking) if compact else None
    before = None if compact else (snapshot_world(world, env.get('audit_ignore_fields')) if offline else None)
    goal_before_state = None
    if offline and tracking is not None and goals:
        try:
            goal_before_state = _goal_state(world, tracking, goals, state)
        except (ValueError, KeyError, TypeError, sqlite3.Error):
            goal_before_state = None
    calls_before = len(world.requester.request_tracker.requests)
    start = time.monotonic()
    output, error = '', None
    try:
        output = world.execute(action_code(action, granularity=env.get('granularity', 'code_block')))
        if not isinstance(output, str):
            raise TypeError('AppWorld.execute must return text')
        if _ERROR.search(output):
            error = 'AppWorld execution reported an error; partial writes are retained'
    except Exception as exception:
        error = f'{type(exception).__name__}: {exception}'
    tracker.max_num_requests = previous_limit
    calls = len(world.requester.request_tracker.requests) - calls_before
    # Code blocks and all API calls are distinct charged resource components.
    cost = 1. + max(calls, 0)
    public = public_state_update(state, action, output, env.get('observations', []))
    if compact:
        audit, before, after = _compact_audit(world, tracking, state, public, before_versions)
    else:
        after = snapshot_world(world, env.get('audit_ignore_fields')) if offline else None
        if before is not None and after is not None:
            before, after = align_audit_states(before, after)
            domain = sorted(set(field_map(before)) | set(field_map(after)))
            changes = semantic_diff(before, after)
        else:
            # Retained public-cache fields can be stale after a side effect. Only
            # fields actually returned by this execution are in a partial audit;
            # cached-but-unread values must never become unchanged negatives.
            fresh = public_state_update({'fields': [], 'relations': []}, action, output, env.get('observations', []))
            domain = sorted(key for key, field in field_map(fresh).items() if field.get('known'))
            changes = sorted(set(semantic_diff(state, public)) & set(domain))
        audit = {'complete': bool(offline), 'field_domain': domain, 'changes': changes,
                 'model_version': 'unselected',
                 'domain_kind': 'global_sqlite' if offline else 'public_partial'}
    result = {'state': public, 'observation': output, 'valid': error is None, 'error': error,
              'cost': cost, 'cost_breakdown': {'code_blocks': 1, 'api_calls': max(calls, 0)}, 'api_calls': max(calls, 0),
              'duration_seconds': time.monotonic() - start,
              'audit': audit,
              'audit_before_state': before, 'audit_state': after, 'fingerprint': _fingerprint(world) if offline else None,
              'done': bool(world.task_completed()), 'evidence_origin': 'appworld'}
    if offline and tracking is not None and goals:
        try:
            goal_after_state = _goal_state(world, tracking, goals, public)
            result.update({
                'goal_before': evaluate_goals(goal_before_state, goals) if goal_before_state is not None else None,
                'goal_after': evaluate_goals(goal_after_state, goals, previous=goal_before_state),
            })
        except (ValueError, KeyError, TypeError, sqlite3.Error) as error:
            result['goal_error'] = f'{type(error).__name__}: {error}'
    return result


def worker_main(connection: Any, request: dict[str, Any]) -> None:
    world = None
    tracking = None
    total_cost = 0.
    pending_cost = False
    try:
        env = request['config']['environment']
        if request['mode'] == 'branch' and request['split'] != 'train':
            raise ValueError('Counterfactual replay is train-only')
        if env.get('offline_audit') and request['split'] not in ('train', 'dev'):
            raise ValueError('Offline audit is train/dev only')
        _set_root(str(Path(env['root']).expanduser().resolve()))
        from appworld import AppWorld
        world = AppWorld(task_id=request['task_id'], experiment_name=request['experiment'],
                         load_ground_truth=False, random_seed=int(request['config'].get('collection', {}).get('seed', 100)),
                         max_interactions=int(env.get('max_interactions', 10000)),
                         max_api_calls_per_interaction=int(env.get('max_api_calls_per_interaction', 1000)),
                         raise_on_unsafe_syntax=True, raise_on_unsafe_execution=True)
        metadata = _public_metadata(world, request)
        state = _initial_state(metadata)
        offline = bool(env.get('offline_audit', False))
        tracking = None
        if offline and env.get('audit_mode', 'compact') == 'compact':
            tracking = _install_change_tracking(world)
            tracking['descriptor'].update({
                'task_id': request['task_id'],
                'db_version': getattr(world.task, 'db_version', None),
                'appworld_revision': env.get('appworld_revision'),
                'app_names': list(world.models.model_names()),
                'task_overrides_sha256': _task_override_digest(str(env.get('root')), request['task_id']),
            })
        goals = metadata.get('goals', [])
        if request['mode'] == 'branch':
            if not offline:
                raise ValueError('Training counterfactuals require offline_audit=true for verified prefix and condition labels')
            total_cost, details = 0., {'replay': 0., 'condition': 0., 'target': 0.}
            prefix_receipts = []
            condition_receipts = []
            def remaining() -> float | None:
                return None if request.get('max_cost') is None else float(request['max_cost']) - total_cost
            def execute_charged(action: dict[str, Any]) -> dict[str, Any]:
                nonlocal pending_cost
                pending_cost = True
                try:
                    result = _execute(world, action, state, env, offline, remaining(),
                                      tracking=tracking, goals=goals)
                except ValueError as error:
                    if 'charged_cost=0' in str(error):
                        pending_cost = False  # Rejected before executing a block.
                    raise
                pending_cost = False
                return result
            for index, entry in enumerate(request.get('prefix', [])):
                action = entry.get('action', entry)
                result = execute_charged(action)
                state = result['state']
                total_cost += result['cost']
                details['replay'] += result['cost']
                prefix_receipts.append({'step': index, 'output': result['observation'], 'fingerprint': result['fingerprint'], 'valid': result['valid'], 'cost': result['cost']})
                # Failed original blocks replay unchanged; partial writes and
                # interpreter locals remain part of their exact prefix.
                if 'output' in entry and result['observation'] != entry['output']:
                    raise ValueError(f'Prefix observation mismatch at step {index}; charged_cost={total_cost}')
                if entry.get('fingerprint') and result['fingerprint'] != entry['fingerprint']:
                    raise ValueError(f'Prefix database fingerprint mismatch at step {index}; charged_cost={total_cost}')
            prefix_state = deepcopy(state)
            prefix_fingerprint = _fingerprint(world)
            condition = request.get('condition')
            if condition:
                validate_condition(condition)
                for intervention in condition.get('actions', []):
                    result = execute_charged(intervention)
                    state = result['state']
                    total_cost += result['cost']
                    details['condition'] += result['cost']
                    condition_receipts.append({'action': deepcopy(intervention), 'output': result['observation'], 'fingerprint': result['fingerprint'], 'valid': result['valid'], 'cost': result['cost']})
                    if not result['valid']:
                        raise ValueError(f'Condition action failed; charged_cost={total_cost}')
                fingerprint = _fingerprint(world)
                result = execute_charged(condition['query'])
                state = result['state']
                total_cost += result['cost']
                details['condition'] += result['cost']
                condition_receipts.append({'action': deepcopy(condition['query']), 'output': result['observation'], 'fingerprint': result['fingerprint'], 'valid': result['valid'], 'cost': result['cost']})
                if not result['valid'] or result['observation'] != condition['expected_output'] or _fingerprint(world) != fingerprint:
                    raise ValueError(f'Condition query failed to establish a read-only observable condition; charged_cost={total_cost}')
            before = deepcopy(state)
            result = execute_charged(request['action'])
            details['target'] = result['cost']
            result.update(before_state=before, prefix_state=prefix_state, prefix_verified=True,
                          prefix_receipts=prefix_receipts, prefix_fingerprint=prefix_fingerprint, condition_receipts=condition_receipts, condition_verified=True, cost=total_cost + result['cost'],
                          cost_breakdown=details, process_isolation='fresh_spawn', process_id=os.getpid())
            connection.send({'ok': True, 'result': result})
            return
        connection.send({'ok': True, 'result': {'state': state, 'metadata': metadata}})
        while True:
            message = connection.recv()
            command = message['command']
            if command == 'close':
                break
            if command == 'execute':
                result = _execute(world, message['action'], state, env, offline,
                                  message.get('budget_remaining'), tracking=tracking, goals=goals)
                state = result['state']
            elif command == 'snapshot':
                if not offline or request['split'] not in ('train', 'dev'):
                    raise ValueError('Global snapshots are offline train/dev only')
                result = {'state': snapshot_world(world, env.get('audit_ignore_fields'))}
            elif command == 'evaluate':
                # Instrumentation must not enter the saved evaluation database.
                _remove_change_tracking(tracking)
                tracking = None
                result = evaluate_fresh(world, request)
            else:
                raise ValueError(f'Unknown worker command {command}')
            connection.send({'ok': True, 'result': result})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as error:
        try:
            connection.send({'ok': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(), 'charged_cost': total_cost, 'cost_unresolved': pending_cost or request.get('mode') != 'branch'})
        except (EOFError, BrokenPipeError):
            pass
    finally:
        _remove_change_tracking(tracking)
        if world is not None:
            world.close()
        connection.close()
