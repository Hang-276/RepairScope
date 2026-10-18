import pytest

from repairscope.schema import evaluate_goals, goal_state_keys, validate_goals


def field(key, value, kind='number', known=True, exists=True, choices=None):
    return {'key': key, 'kind': kind, 'value': value, 'known': known, 'exists': exists,
            'choices': choices or []}


def state(*fields):
    return {'fields': list(fields), 'relations': []}


def test_backward_compatible_single_field_goal():
    goals = validate_goals([{'id': 'g', 'field': 'a.b[1].x', 'op': 'eq', 'value': 5}])
    assert evaluate_goals(state(field('a.b[1].x', 5)), goals) == [True]
    assert evaluate_goals(state(field('a.b[1].x', 4)), goals) == [False]
    assert evaluate_goals(state(field('a.b[1].x', None, known=False)), goals) == [None]


def test_all_any_none_with_wildcards_and_unknown():
    fields = [field('spotify.songs[1].rating', 5), field('spotify.songs[2].rating', 4)]
    universe = state(*fields)
    all_five = [{'id': 'g', 'kind': 'all', 'field': 'spotify.songs[*].rating', 'op': 'eq', 'value': 5}]
    assert evaluate_goals(universe, all_five) == [False]
    assert evaluate_goals(state(field('spotify.songs[1].rating', 5), field('spotify.songs[2].rating', 5)),
                          all_five) == [True]
    unknown = state(field('spotify.songs[1].rating', 5), field('spotify.songs[2].rating', None, known=False))
    assert evaluate_goals(unknown, all_five) == [None]
    any_five = [{'id': 'g', 'kind': 'any', 'field': 'spotify.songs[*].rating', 'op': 'eq', 'value': 5}]
    assert evaluate_goals(universe, any_five) == [True]
    assert evaluate_goals(state(field('spotify.songs[1].rating', 4), field('spotify.songs[2].rating', 3)),
                          any_five) == [False]
    none_five = [{'id': 'g', 'kind': 'none', 'field': 'spotify.songs[*].rating', 'op': 'eq', 'value': 5}]
    assert evaluate_goals(universe, none_five) == [False]
    assert evaluate_goals(state(field('spotify.songs[1].rating', 4)), none_five) == [True]


def test_count_and_explicit_fields():
    fields = [field('a.t[1].x', 5), field('a.t[2].x', 5), field('a.t[3].x', 4)]
    goal = [{'id': 'g', 'kind': 'count', 'fields': ['a.t[1].x', 'a.t[2].x', 'a.t[3].x'],
             'filter_op': 'eq', 'filter_value': 5, 'op': 'ge', 'value': 2}]
    assert evaluate_goals(state(*fields), goal) == [True]
    goal[0]['value'] = 3
    assert evaluate_goals(state(*fields), goal) == [False]
    # Unknown values may satisfy the missing count but cannot prove it.
    goal[0]['value'] = 2
    assert evaluate_goals(state(fields[0], field('a.t[2].x', None, known=False)), goal) == [None]
    goal[0]['value'] = 3
    assert evaluate_goals(state(fields[0], field('a.t[2].x', None, known=False)), goal) == [False]
    count_all = [{'id': 'g', 'kind': 'count', 'field': 'a.t[*].x', 'op': 'eq', 'value': 3}]
    assert evaluate_goals(state(*fields), count_all) == [True]


@pytest.mark.parametrize('aggregate,value,expected', [
    ('sum', 9, True), ('mean', 4.5, True), ('min', 4, True), ('max', 5, True), ('count', 2, True),
])
def test_aggregate_goals(aggregate, value, expected):
    fields = state(field('a.t[1].x', 5), field('a.t[2].x', 4))
    goal = [{'id': 'g', 'kind': 'aggregate', 'aggregate': aggregate,
             'field': 'a.t[*].x', 'op': 'eq', 'value': value}]
    assert evaluate_goals(fields, goal) == [expected]


def test_computed_per_object_target():
    previous = state(field('a.t[1].x', 5), field('a.t[2].x', 4))
    goal = [{'id': 'g', 'kind': 'all', 'field': 'a.t[*].x', 'op': 'eq', 'expr': 'previous_value + 1'}]
    assert evaluate_goals(state(field('a.t[1].x', 6), field('a.t[2].x', 5)), goal, previous=previous) == [True]
    assert evaluate_goals(state(field('a.t[1].x', 5), field('a.t[2].x', 4)), goal, previous=previous) == [False]


def test_answer_and_complete_goals():
    answer = [{'id': 'g', 'kind': 'answer', 'value': 'blue'}]
    assert evaluate_goals(state(field('supervisor.tasks[1].answer', 'blue', 'text')), answer) == [True]
    assert evaluate_goals(state(field('supervisor.tasks[1].answer', 'red', 'text')), answer) == [False]
    complete = [{'id': 'g', 'kind': 'complete'}]
    assert evaluate_goals(state(field('supervisor.tasks[1].status', 'success', 'text')), complete) == [True]
    assert evaluate_goals(state(field('supervisor.tasks[1].status', 'fail', 'text')), complete) == [False]


def test_goal_state_keys_matches_patterns():
    goal = {'id': 'g', 'kind': 'all', 'field': 'spotify.songs[*].rating', 'op': 'eq', 'value': 5}
    keys = goal_state_keys(goal, state(field('spotify.songs[1].rating', 5),
                                       field('spotify.songs[2].rating', 5),
                                       field('spotify.playlists[1].rating', 5)))
    assert keys == ['spotify.songs[1].rating', 'spotify.songs[2].rating']


def test_goal_validation_rejects_unsafe_expression_and_unknown_keys():
    with pytest.raises(ValueError, match='forbids'):
        validate_goals([{'id': 'g', 'field': 'a.b[1].x', 'op': 'eq', 'expr': '__import__("os")'}])
    with pytest.raises(ValueError, match='unsupported keys'):
        validate_goals([{'id': 'g', 'field': 'a.b[1].x', 'op': 'eq', 'value': 1, 'evil': 1}])
