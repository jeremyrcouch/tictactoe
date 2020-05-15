import json
import numpy as np
import pytest

from table import (initialize_value_map, format_value_map,
    state_lookup, collect_values, TablePlayer, ValueMod)
from utils.helpers import Game, str_to_tuple, tuple_to_str
from utils.players import MoveRecord


DATA_PATH = 'tests/data/'


@pytest.fixture(scope='module')
def value_map():
    with open('{}init_value_map.json'.format(DATA_PATH), 'r') as fp:
        load_map = json.load(fp)
    return format_value_map(load_map, str_to_tuple)
    # return initialize_value_map(INITIAL_VALUE)


def test_initialize_value_map(value_map):
    # arrange
    # act
    # assert
    assert isinstance(value_map, dict)


def test_format_value_map(value_map):
    # arrange
    # act
    mod_map = format_value_map(value_map, tuple_to_str)
    unmod_map = format_value_map(mod_map, str_to_tuple)

    # assert
    assert len(value_map) == len(mod_map)
    assert len(unmod_map) == len(value_map)
    assert unmod_map == value_map


def test_state_lookup(value_map):
    # arrange
    state = np.reshape((-1, 1, 0, 1, 0, -1, 0, 0, 0), Game.board_shape)

    # act
    match_state, transform = state_lookup(state, value_map)

    # assert
    assert isinstance(match_state, tuple)
    assert isinstance(transform, dict)
    assert len(match_state) == (Game.board_shape[0]*Game.board_shape[1])
    assert 'func' in transform
    assert 'args' in transform


def test_state_lookup_no_match(value_map):
    # arrange
    state = np.reshape((-1, -1, -1, -1, -1, -1, -1, -1, -1), Game.board_shape)

    # act + assert
    with pytest.raises(ValueError):
        _ = state_lookup(state, value_map)


@pytest.mark.parametrize(
    "terminal, expected",
    [
        pytest.param(True, 7, id="include-terminal"),
        pytest.param(False, 5, id="no-terminal")
    ],
)
def test_collect_values(terminal, expected):
    # arrange
    val = 0.5
    value_map = {
        (0, 0): {1: {(0, 0): val, (0, 1): val}, -1: {(0, 0): val, (0, 1): val}},
        (1, 0): {1: {}, -1: {(0, 1): val}},
        (1, -1): {1: val, -1: val}
    }

    # act
    values = collect_values(value_map, include_terminal=terminal)

    # assert
    assert len(values) == expected


def test_TablePlayer_play(value_map):
    # arrange
    player = TablePlayer(value_map)
    marker = 1
    game = Game()

    # act
    loc = player.play(marker, game)

    # assert
    assert isinstance(loc, tuple)


@pytest.mark.skip(reason='_policy == play for TablePlayer')
def test_TablePlayer_policy():
    pass


@pytest.mark.parametrize(
    "reward, expected",
    [
        pytest.param(0, 'eq', id="zero-reward"),
        pytest.param(1, 'pos', id="positive"),
        pytest.param(-1, 'neg', id="negative")
    ],
)
def test_TablePlayer_process_reward(value_map, reward, expected):
    # arrange
    player = TablePlayer(value_map)
    marker = 1
    player.buffer = [
        MoveRecord(state=np.reshape((0, -1, -1, 0, 0, 1, 0, 0, 1), Game.board_shape),
                   move=(0, 0),
                   marker=marker),
        MoveRecord(state=np.reshape((1, -1, -1, 0, 0, 1, -1, 0, 1), Game.board_shape),
                   move=(1, 1),
                   marker=marker)
    ]

    # act
    value_mods = player.process_reward(reward, Game.ind_to_loc)

    # assert
    if expected == 'eq':
        assert value_mods is None
    else:
        assert len(value_mods) == len(player.buffer)
        if expected == 'pos':
            assert all([vm.new >= vm.previous for vm in value_mods])
        elif expected == 'neg':
            assert all([vm.new <= vm.previous for vm in value_mods])
