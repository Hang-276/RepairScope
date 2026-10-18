"""Persistent, separate environment and Actor caps for recorded acquisition.

The lock covers a complete collection command.  Natural collection and every
counterfactual command use the same ledger, so starting a new phase never grants
a new allowance.  Only cost receipts and operation metadata enter this file.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
from typing import Any


class AcquisitionBudgetExceeded(RuntimeError):
    """An external call was blocked before it could exceed its allowance."""


class CollectionBudgetError(RuntimeError):
    """A ledger is inconsistent, unresolved, or cannot guarantee a hard cap."""


_LIMIT_KEYS = (
    'environment_limit', 'natural_step_limit', 'cf_cell_limit',
    'actor_generation_limit', 'actor_http_limit',
)
_RESOURCE_LIMIT = {
    'environment': 'environment_limit',
    'actor_generations': 'actor_generation_limit',
    'actor_http_attempts': 'actor_http_limit',
}
_EPSILON = 1e-9
_NUMBER = r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
_CHARGED = re.compile(r'(?:^|[;\s])charged_cost\s*[=:]\s*(' + _NUMBER + r')(?=$|[;\s])')
_UNRESOLVED = re.compile(r'(?:^|[;\s])cost_unresolved\s*[=:]\s*true(?=$|[;\s])', re.I)


def _cost(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _time() -> str:
    return datetime.now(timezone.utc).isoformat()


class CollectionBudget:
    """One acquisition transaction with persistent cross-command allowances.

    AppWorld requires an explicit ``collection.budget`` mapping.  Fixture runs
    without one keep the previous in-memory, uncapped behavior.  ``run_cost`` is
    the known environment cost incurred by this invocation, not lifetime spend;
    an unresolved operation may have an additional, deliberately unknown cost.
    """

    def __init__(self, config: Mapping[str, Any]):
        settings = config.get('collection', {})
        budget = settings.get('budget')
        self.active = budget is not None
        self.run_cost = 0.0
        self._entered = False
        self._lock = None
        self.path: Path | None = None
        self._limits: dict[str, int | float] = {}
        self._run_id: str | None = None
        if not self.active:
            if config.get('environment', {}).get('backend', 'appworld') != 'fixture':
                raise CollectionBudgetError('AppWorld acquisition requires explicit collection.budget hard limits')
        else:
            if not isinstance(budget, Mapping):
                raise CollectionBudgetError('collection.budget must be a mapping')
            missing = [key for key in ('ledger_path', 'run_id', *_LIMIT_KEYS) if key not in budget]
            if missing:
                raise CollectionBudgetError('collection.budget requires explicit fields: ' + ', '.join(missing))
            try:
                self.path = Path(budget['ledger_path']).expanduser()
            except (TypeError, ValueError) as error:
                raise CollectionBudgetError('collection.budget.ledger_path must be an absolute path') from error
            if not self.path.is_absolute():
                raise CollectionBudgetError('collection.budget.ledger_path must be an absolute path')
            self.path = self.path.resolve()
            self._run_id = budget['run_id']
            if not isinstance(self._run_id, str) or not self._run_id.strip():
                raise CollectionBudgetError('collection.budget.run_id must be a nonempty string')
            for key in _LIMIT_KEYS:
                value = _cost(budget[key])
                if value is None:
                    raise CollectionBudgetError(f'collection.budget.{key} must be finite and nonnegative')
                if key.startswith('actor_'):
                    if type(budget[key]) is not int:
                        raise CollectionBudgetError(f'collection.budget.{key} must be an integer count')
                    self._limits[key] = budget[key]
                else:
                    self._limits[key] = value
        self._data: dict[str, Any] = {
            'format_version': 1, 'run_id': self._run_id, 'limits': deepcopy(self._limits),
            'spent': {'environment': 0.0, 'actor_generations': 0, 'actor_http_attempts': 0},
            'pending': None, 'cost_unresolved': False, 'sealed': False,
            'seal_reason': None, 'events': [],
        }

    @property
    def cost_unresolved(self) -> bool:
        return self._data['cost_unresolved']

    def __enter__(self) -> 'CollectionBudget':
        if self._entered:
            raise CollectionBudgetError('CollectionBudget is already open')
        self.run_cost = 0.0
        try:
            if self.active:
                assert self.path is not None
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._lock = self.path.with_name(self.path.name + '.lock').open('a+', encoding='utf-8')
                try:
                    fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise CollectionBudgetError('Acquisition ledger is already in use; no external call was made') from error
                if self.path.exists():
                    try:
                        loaded = json.loads(self.path.read_text(encoding='utf-8'))
                    except (OSError, ValueError) as error:
                        raise CollectionBudgetError('Acquisition ledger is unreadable; refusing to reset it') from error
                    self._validate_loaded(loaded)
                    self._data = loaded
                else:
                    self._persist()
                if self._data['pending'] is not None:
                    self._seal('unresolved_pending_operation', unresolved=True)
                    self._persist()
                if self._data['cost_unresolved'] and not self._data['sealed']:
                    self._seal('unresolved_cost', unresolved=True)
                    self._persist()
                if self._data['sealed']:
                    raise CollectionBudgetError('Acquisition ledger is sealed: ' + str(self._data['seal_reason']))
            self._entered = True
            return self
        except BaseException:
            self._release()
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._entered = False
        self._release()

    def _release(self) -> None:
        if self._lock is not None:
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock.close()
                self._lock = None

    def _validate_loaded(self, data: Any) -> None:
        if not isinstance(data, dict) or data.get('format_version') != 1:
            raise CollectionBudgetError('Unsupported acquisition ledger; refusing to reset it')
        if data.get('run_id') != self._run_id or data.get('limits') != self._limits:
            raise CollectionBudgetError('Acquisition run_id and limits are immutable for an existing ledger')
        spent = data.get('spent')
        if not isinstance(spent, dict) or set(spent) != set(_RESOURCE_LIMIT):
            raise CollectionBudgetError('Invalid acquisition ledger counters')
        for resource, limit in _RESOURCE_LIMIT.items():
            value = _cost(spent[resource])
            if value is None or (resource != 'environment' and type(spent[resource]) is not int):
                raise CollectionBudgetError('Invalid acquisition ledger counters')
            if value > self._limits[limit] + _EPSILON and data.get('sealed') is not True:
                raise CollectionBudgetError('Acquisition ledger exceeds its limits without a seal')
        if (type(data.get('sealed')) is not bool or type(data.get('cost_unresolved')) is not bool
                or not isinstance(data.get('events'), list) or 'pending' not in data
                or (data['pending'] is not None and not isinstance(data['pending'], dict))):
            raise CollectionBudgetError('Invalid acquisition ledger state')

    def _persist(self) -> None:
        if not self.active:
            return
        assert self.path is not None
        temporary = self.path.with_name(self.path.name + '.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(self._data, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _ensure_open(self) -> None:
        if not self._entered:
            raise CollectionBudgetError('Use CollectionBudget as a context manager')
        if self._data['sealed'] or self._data['cost_unresolved'] or self._data['pending'] is not None:
            raise CollectionBudgetError('Acquisition ledger is sealed or has an unresolved operation')

    def _remaining(self, resource: str) -> float | None:
        if not self.active:
            return None
        return max(0.0, self._limits[_RESOURCE_LIMIT[resource]] - self._data['spent'][resource])

    def snapshot(self) -> dict[str, Any]:
        return {**deepcopy(self._data), 'active': self.active,
                'ledger_path': str(self.path) if self.path is not None else None,
                'remaining': {resource: self._remaining(resource) for resource in _RESOURCE_LIMIT},
                'run_cost': self.run_cost}

    def _seal(self, reason: str, *, unresolved: bool = False) -> None:
        self._data.update(sealed=True, seal_reason=reason)
        self._data['cost_unresolved'] = self._data['cost_unresolved'] or unresolved

    def _event(self, **fields: Any) -> dict[str, Any]:
        event = {'id': len(self._data['events']) + 1, 'time_utc': _time(), **fields}
        self._data['events'].append(event)
        return event

    def _settle_environment(self, event: dict[str, Any], cost: float | None,
                            *, failed: bool, unresolved: bool, cap: float | None) -> None:
        event['status'] = 'unresolved' if unresolved else 'failed' if failed else 'completed'
        event['cost_unresolved'] = unresolved
        if cost is not None and (not unresolved or cost > 0):
            event['known_cost_lower_bound' if unresolved else 'charged_cost'] = cost
            self._data['spent']['environment'] += cost
            self.run_cost += cost
        event['cumulative_environment_cost'] = self._data['spent']['environment']
        if unresolved:
            self._seal('unresolved_environment_cost', unresolved=True)
        else:
            self._data['pending'] = None
        if cost is not None and cap is not None and cost > cap + _EPSILON:
            self._seal('environment_cost_overrun', unresolved=unresolved)
        self._persist()

    @staticmethod
    def _failed_cost(error: BaseException) -> tuple[float | None, bool]:
        text = str(error)
        unresolved = bool(getattr(error, 'cost_unresolved', False)) or bool(_UNRESOLVED.search(text))
        value = _cost(getattr(error, 'charged_cost', None))
        if value is None:
            matches = _CHARGED.findall(text)
            if len(matches) == 1:
                value = _cost(float(matches[0]))
        return value, unresolved or value is None

    def environment_cap(self, kind: str) -> float | None:
        """Check affordability before, for example, asking an Actor for a move."""
        self._ensure_open()
        if kind not in ('natural', 'cf'):
            raise ValueError('Acquisition environment kind must be natural or cf')
        cap = None
        if self.active:
            cap = min(self._remaining('environment'),
                      self._limits['natural_step_limit' if kind == 'natural' else 'cf_cell_limit'])
            if cap < 1:
                raise AcquisitionBudgetExceeded('Environment allowance exhausted before ' + kind)
        return cap

    def environment_call(self, kind: str, call: Callable[[float | None], Any]) -> Any:
        cap = self.environment_cap(kind)
        event = self._event(resource='environment', kind=kind, status='pending', cap=cap)
        self._data['pending'] = {'event_id': event['id'], 'kind': kind, 'cap': cap}
        self._persist()
        try:
            result = call(cap)
        except BaseException as error:
            cost, unresolved = self._failed_cost(error)
            event['error_type'] = type(error).__name__
            self._settle_environment(event, cost, failed=True, unresolved=unresolved, cap=cap)
            raise
        cost = _cost(result.get('cost')) if isinstance(result, Mapping) else None
        unresolved = cost is None or bool(result.get('cost_unresolved', False))
        self._settle_environment(event, cost, failed=False, unresolved=unresolved, cap=cap)
        if unresolved:
            raise CollectionBudgetError('Environment returned no resolved finite actual cost; acquisition ledger sealed')
        if self._data['sealed']:
            raise AcquisitionBudgetExceeded('Environment exceeded its hard cap; actual cost recorded and ledger sealed')
        return result

    def _charge_actor(self, resource: str) -> None:
        self._ensure_open()
        remaining = self._remaining(resource)
        if remaining is not None and remaining < 1:
            raise AcquisitionBudgetExceeded(resource + ' allowance exhausted before external call')
        self._data['spent'][resource] += 1
        self._event(resource=resource, kind=resource, status='charged', units=1,
                    cumulative_cost=self._data['spent'][resource])
        self._persist()

    def actor_generate(self, actor: Any, call: Callable[[], Any]) -> Any:
        self._ensure_open()
        setter = getattr(actor, 'set_http_budget_hook', None)
        if self.active and not callable(setter):
            raise CollectionBudgetError('Actor must support set_http_budget_hook for pre-request hard limits')
        installed = callable(setter)
        previous = None
        def charge_http():
            if previous is not None:
                previous()
            self._charge_actor('actor_http_attempts')
        previous = setter(charge_http) if installed else None
        try:
            self._charge_actor('actor_generations')
            return call()
        finally:
            if installed:
                setter(previous)
