from copy import deepcopy
from pathlib import Path

from repairscope.actor import FrozenActor, normalize_action
from repairscope.collection import _align_counterfactual_audits
from repairscope.config import load_config
from repairscope.control import RepairController
from repairscope.schema import field_map, validate_action, validate_transition


def _field(key, value, kind='text'):
    return {'key': key, 'kind': kind, 'value': value, 'known': True, 'exists': True, 'choices': []}


def _state(*fields):
    return {'fields': list(fields), 'relations': []}


def _cf_record(role, condition='c0', fields=(), changes=()):
    state = _state(*fields)
    return {
        'task_id': 'task-1', 'group_id': 'task-1', 'source_task_id': 'task-1',
        'source_split': 'train', 'split': 'train', 'trajectory_id': 'task-1-cf',
        'step': 1, 'state': _state(*fields), 'action': {'name': 'a.b', 'arguments': {}, 'cost': 1},
        'next_state': deepcopy(state), 'history': [], 'goals': [],
        'goal_before': [], 'goal_after': [],
        'scope': [], 'audit': {'complete': True, 'field_domain': [f['key'] for f in fields],
                               'changes': list(changes), 'model_version': 'm',
                               'domain_kind': 'compact_sqlite'},
        'audit_state': deepcopy(state), 'audit_next_state': deepcopy(state),
        'action_valid': True, 'execution_cost': 1, 'evidence_origin': 'appworld',
        'used_for_training': True, 'cf_pair_id': 'pair-1', 'cf_condition': condition,
        'cf_role': role, 'prefix_verified': True, 'condition_verified': True,
    }


def test_normalize_action_drops_empty_code():
    action = normalize_action({'name': 'a.b', 'arguments': {}, 'code': '   ', 'cost': 1})
    assert 'code' not in action
    validate_action(action)


def test_actor_stats_shape():
    actor = FrozenActor({'kind': 'scripted', 'actions': [{'name': 'a.b', 'arguments': {}}]})
    actor.generate({'fields': [], 'relations': []}, [], [])
    stats = actor.stats()
    assert stats['calls'] == 1
    assert stats['http_calls'] == 0
    assert stats['retries'] == 0


class _RetryingActor:
    last_http_calls = 3

    def generate(self, *args, **kwargs):
        return [{'name': 'noop', 'arguments': {}, 'cost': 1}]


def test_actor_http_retries_are_charged():
    controller = RepairController(_RetryingActor(), None,
                                  {'candidate_count': 1, 'max_calls': 10, 'actor_cost': 1})
    controller._candidates(_state(), [], 0, 'execute', '')
    assert controller.budget.spent == 3
    kinds = [event['kind'] for event in controller.budget.events]
    assert 'actor' in kinds and 'actor_retry_adjustment' in kinds


def test_primary_variants_share_actor_and_budget():
    path = Path(__file__).resolve().parents[2] / 'configs' / 'repairscope' / 'base.yaml'
    a1 = load_config(path, variant='dynamics')
    a3 = load_config(path, variant='full')
    for section in ('actor', 'model', 'train', 'collection', 'evaluation', 'probing', 'environment'):
        assert a1[section] == a3[section], section
    assert a1['loss']['boundary'] == 0 and a1['loss']['escape'] == 0
    assert a3['loss']['boundary'] == 1 and a3['loss']['escape'] == 1


def test_counterfactual_domain_alignment_keeps_values_and_changes():
    field = _field('supervisor.tasks[1].status', 'success')
    action = _cf_record('action', fields=(field,), changes=(field['key'],))
    reference = _cf_record('reference')
    for record in (action, reference):
        validate_transition(record)
    _align_counterfactual_audits([action, reference])
    for record in (action, reference):
        validate_transition(record)
        assert field['key'] in record['audit']['field_domain']
        assert field['key'] in field_map(record['audit_state'])
        assert field_map(record['audit_state'])[field['key']]['value'] == 'success'
        assert field_map(record['audit_next_state'])[field['key']]['value'] == 'success'
    assert action['audit']['changes'] == [field['key']]
    assert reference['audit']['changes'] == []


def test_compact_audit_does_not_false_reject_matched_prefix():
    from repairscope.losses import _matched_prefix
    common = _field('supervisor.tasks[1].status', 'pending')
    extra = _field('supervisor.tasks[1].answer', 'x')
    left = _cf_record('action', fields=(common, extra))
    right = _cf_record('reference', fields=(common,))
    # Public prefix states are identical; only the compact audit stores differ.
    right['state'] = deepcopy(left['state'])
    right['next_state'] = deepcopy(left['next_state'])
    records = [left, right]
    assert _matched_prefix(records, 0, 1)
    changed = _field('supervisor.tasks[1].status', 'other')
    right2 = _cf_record('reference', fields=(changed,))
    right2['state'] = deepcopy(left['state'])
    right2['next_state'] = deepcopy(left['next_state'])
    assert not _matched_prefix([left, right2], 0, 1)


def test_candidate_legality_rejects_placeholders_and_name_mismatch():
    from repairscope.actor import candidate_is_legal
    good = {'name': 'spotify.show_playlist', 'arguments': {},
            'code': "login = apis.spotify.login(username='a', password='b')\n"
                    "print(apis.spotify.show_playlist(access_token=login['access_token'], playlist_id=1))"}
    assert candidate_is_legal(good)
    bad_token = {'name': 'spotify.show_playlist', 'arguments': {},
                 'code': "print(apis.spotify.show_playlist(access_token='obtained_from_supervisor_profile'))"}
    assert not candidate_is_legal(bad_token)
    bad_name = {'name': 'supervisor.complete_task', 'arguments': {},
                'code': "print(apis.simple_note.show_note(note_id=1))"}
    assert not candidate_is_legal(bad_name)
    no_apis = {'name': 'a.b', 'arguments': {}, 'code': "print('hello')"}
    assert not candidate_is_legal(no_apis)
