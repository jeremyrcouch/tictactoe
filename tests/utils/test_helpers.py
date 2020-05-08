import numpy as np
import pytest

from src.utils.helpers import (Game, array_in_list, moving_average, state_to_actions,
    checkstates, state_transforms, reverse_transforms, reverse_function)


@pytest.mark.parametrize(
    "arr, arr_list, expected",
    [
        pytest.param([0, 1, 2], [], False, id="empty-list"),
        pytest.param([0, 1, 2], [[2, 1, 0], [], [0, -1, 2]], False, id="not-in"),
        pytest.param([[0, 1], [2, 3]], [[1, 1], [[0, 1], [2, 3]]], True, id="in"),
    ],
)
def test_array_in_list(arr, arr_list, expected):
    # arrange
    arr_in = np.array(arr)
    arr_list_in = [np.array(a) for a in arr_list]

    # act
    is_in = array_in_list(arr_in, arr_list_in)

    # assert
    assert expected == is_in


@pytest.mark.parametrize(
    "vals, n, expected",
    [
        pytest.param([0, 1, 2, 3, 4], 10, [], id="n>len(vals)"),
        pytest.param([0, 1, 2, 3, 4], 3, [1, 2, 3], id="normal"),
    ],
)
def test_moving_average(vals, n, expected):
    # arrange
    expected_length = (len(vals) - (n - 1)) if n < len(vals) else 0

    # act
    ma = moving_average(vals, n=n)

    # assert
    assert len(ma) == expected_length
    assert np.array_equal(ma, np.array(expected))


def test_moving_average_invalid_n():
    # arrange
    n = 0
    vals = [1, 2, 3]

    # act + assert
    with pytest.raises(ValueError):
        _ = moving_average(vals, n)


def test_state_to_actions():
    # arrange
    state = (0, 1, -1, 1, 0, -1, -1, 1, 0)
    expected_actions = [(0, 0), (1, 1), (2, 2)]

    # act
    actions = state_to_actions(state, Game.ind_to_loc, Game.empty_marker)

    # assert
    assert set(actions) == set(expected_actions)


def test_check_states():
    # arrange
    state = np.reshape((0, 1, -1, 0, 0, -1, -1, 1, 1), Game.board_shape)
    expected_count = 12
    expected_transforms = [
        {'func': None, 'args': {}},
        {'func': np.rot90, 'args': {'k': -1}},
        {'func': np.rot90, 'args': {'k': -2}},
        {'func': np.rot90, 'args': {'k': -3}},
        {'func': np.fliplr, 'args': {}},
        {'func': np.flipud, 'args': {}}
    ]
    expected_states = {
        (0, 1, -1, 0, 0, -1, -1, 1, 1),
        (-1, 1, 1, 0, 0, -1, 0, 1, -1),
        (-1, 1, 0, -1, 0, 0, 1, 1, -1),
        (-1, -1, 1, 1, 0, 1, 0, 0, -1),
        (1, 1, -1, -1, 0, 0, -1, 1, 0),
        (-1, 0, 0, 1, 0, 1, 1, -1, -1),
        (0, -1, 1, 0, 0, 1, 1, -1, -1),
        (1, -1, -1, 0, 0, 1, 0, -1, 1),
        (1, -1, 0, 1, 0, 0, -1, -1, 1),
        (1, 1, -1, -1, 0, -1, 0, 0, 1),
        (-1, -1, 1, 1, 0, 0, 1, -1, 0),
        (1, 0, 0, -1, 0, -1, -1, 1, 1)
    }

    # act
    states, transforms = checkstates(state)

    # assert
    assert len(states) == expected_count
    assert len(transforms) == expected_count
    assert set(states) == expected_states
    assert all([t in transforms for t in expected_transforms])


def test_state_transforms():
    # arrange
    state = (0, 1, -1, 0, 0, -1, -1, 1, 1)
    expected_count = 6
    expected_transforms = [
        {'func': None, 'args': {}},
        {'func': np.rot90, 'args': {'k': -1}},
        {'func': np.rot90, 'args': {'k': -2}},
        {'func': np.rot90, 'args': {'k': -3}},
        {'func': np.fliplr, 'args': {}},
        {'func': np.flipud, 'args': {}}
    ]
    expected_states = {
        (0, 1, -1, 0, 0, -1, -1, 1, 1),
        (-1, 1, 1, 0, 0, -1, 0, 1, -1),
        (-1, 1, 0, -1, 0, 0, 1, 1, -1),
        (-1, -1, 1, 1, 0, 1, 0, 0, -1),
        (1, 1, -1, -1, 0, 0, -1, 1, 0),
        (-1, 0, 0, 1, 0, 1, 1, -1, -1)
    }

    # act
    states, transforms = state_transforms(state)

    # assert
    assert len(states) == expected_count
    assert len(transforms) == expected_count
    assert set(states) == expected_states
    assert all([t in transforms for t in expected_transforms])


def test_reverse_transform():
    # arrange
    action_values = {
        (0, 0): 0,
        (0, 1): 0.1,
        (0, 2): 0.2,
        (1, 0): 0.3,
        (1, 1): 0.4,
        (2, 2): 0.5,
        (2, 0): 0.6,
        (2, 1): 0.7,
        (2, 2): 0.8
    }
    transform = {'func': np.fliplr, 'args': {}}
    expected_values = [action_values[act] for act in action_values]

    # act
    adj_values = reverse_transforms(action_values, transform, Game.ind_to_loc)
    values = [adj_values[act] for act in adj_values]
    print(adj_values)

    # assert
    assert len(adj_values) == len(action_values)
    assert set(values) == set(expected_values)


@pytest.mark.parametrize(
    "loc, func, func_args, expected_loc",
    [
        pytest.param((0, 1), None, {}, (0, 1), id="none"),
        pytest.param((0, 1), np.rot90, {'k': -1}, (1, 2), id="rot90"),
        pytest.param((0, 1), np.rot90, {'k': -2}, (2, 1), id="rot180"),
        pytest.param((0, 1), np.rot90, {'k': -3}, (1, 0), id="rot270"),
        pytest.param((0, 1), np.fliplr, {}, (0, 1), id="fliplr"),
        pytest.param((0, 1), np.flipud, {}, (2, 1), id="flipud"),
    ],
)
def test_reverse_function(loc, func, func_args, expected_loc):
    # arrange
    # act
    new_loc = reverse_function(loc, Game.ind_to_loc, func, func_args)

    # assert
    assert new_loc == expected_loc