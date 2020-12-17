import numpy as np
import pytest

import torch
import torch.nn as nn

from neural import (ValueMod, linear_net, one_hot_state, NeuralPlayer)
from utils.helpers import Game, state_to_actions
from utils.players import MoveRecord


DATA_PATH = 'tests/data/'

@pytest.fixture
def net():
    return linear_net(Game())


def test_linear_net(net):
    assert isinstance(net, nn.Sequential)


def test_one_hot_state():
    game = Game()
    state = np.reshape((0, 1, -1, 0, 1, 0, -1, 0, 0), game.board_shape)
    marker_order = [-1, 0, 1]
    expected_size = len(marker_order)*state.size
    expected_ohe = np.array(
        [0, 0, 1, 0, 0, 0, 1, 0, 0,
         1, 0, 0, 1, 0, 1, 0, 1, 1,
         0, 1, 0, 0, 1, 0, 0, 0, 0],
        dtype=np.int8
    )

    ohe = one_hot_state(state, marker_order)

    assert ohe.size == expected_size
    assert (ohe == expected_ohe).all()


@pytest.mark.skip(reason='visual')
def test_show_move_values():
    pass


def test_NeuralPlayer_state_values(net):
    game = Game()
    lr = 0.25
    agent = NeuralPlayer(net, lr)
    state = np.reshape((0, 1, -1, 0, 1, 0, -1, 0, 0), game.board_shape)
    expected_len = game.board_shape[0]*game.board_shape[1]

    values = agent._state_values(state)

    assert isinstance(values, torch.Tensor)
    assert len(values) == expected_len


@pytest.mark.parametrize(
    "marker, expected",
    [
        pytest.param(1, [0, 1, -1, 0, 1, 0, -1, 0, 0], id="no-swap"),
        pytest.param(-1, [0, -1, 1, 0, -1, 0, 1, 0, 0], id="swap")
    ],
)
def test_NeuralPlayer_adjust_state_for_marker(net, marker, expected):
    game = Game()
    lr = 0.25
    agent = NeuralPlayer(net, lr)
    state = np.reshape((0, 1, -1, 0, 1, 0, -1, 0, 0), game.board_shape)
    expected_mod = np.reshape(expected, game.board_shape)

    state_mod = agent._adjust_state_for_marker(state, marker)

    assert (state_mod == expected_mod).all()


def test_NeuralPlayer_policy(net):
    game = Game()
    lr = 0.25
    agent = NeuralPlayer(net, lr)
    marker = 1
    state = np.reshape((0, 1, -1, 0, 1, 0, -1, 0, 0), game.board_shape)
    game.state = state

    move_values = agent._policy(marker, game)

    assert isinstance(move_values, list)


@pytest.mark.parametrize(
    "value, reward",
    [
        pytest.param(0.01, -1, id="less-than-zero"),
        pytest.param(0.99, 1, id="greater-than-one"),
        pytest.param(0.5, 0, id="no-reward"),
        pytest.param(0.5, 1, id="win"),
        pytest.param(0.5, -1, id="lose")
    ],
)
def test_NeuralPlayer_update_value_with_reward(net, value, reward):
    game = Game()
    lr = 0.25
    agent = NeuralPlayer(net, lr)
    temp_disc = 0.5

    updated = agent._update_value_with_reward(value, reward, lr, temp_disc)

    assert updated >= 0
    assert updated <= 1
    if reward == 0:
        assert updated == value
    elif reward > 0:
        assert updated > value
    elif reward < 0:
        assert updated < value


def test_NeuralPlayer_calc_target_values(net):
    game = Game()
    lr = 0.25
    agent = NeuralPlayer(net, lr)
    n_vals = game.board_shape[0]*game.board_shape[1]
    rand_vals = list(np.random.rand(n_vals, 1))
    values = torch.tensor(rand_vals, dtype=float)
    move_ind = 5
    valid_inds = [1, 2, 3]
    current = values[move_ind].item()
    updated = current*1.1

    targets = agent._calc_target_values(values, current, updated, move_ind, valid_inds)

    assert np.isclose(torch.sum(targets).item(), 1)


def test_NeuralPlayer_play(net):
    game = Game()
    lr = 0.25
    agent = NeuralPlayer(net, lr)
    marker = 1
    state = np.reshape((0, 1, -1, 0, 1, 0, -1, 0, 0), game.board_shape)
    game.state = state
    actions = state_to_actions(tuple(state.flatten()), game.ind_to_loc, game.empty_marker)

    loc = agent.play(marker, game)

    assert isinstance(loc, tuple)
    assert loc in actions


def test_NeuralPlayer_equivalent_states_to_reward(net):
    game = Game()
    lr = 0.25
    agent = NeuralPlayer(net, lr)
    state = np.reshape((0, 1, -1, 0, 1, 0, -1, 0, 0), game.board_shape)

    equiv_states, equiv_transforms = agent._equivalent_states_to_reward(state)

    assert len(equiv_states) == len(equiv_transforms)


def test_NeuralPlayer_reward_move(net):
    game = Game()
    lr = 0.25
    agent = NeuralPlayer(net, lr)
    state = np.reshape((0, 1, -1, 0, 1, 0, -1, 0, 0), game.board_shape)
    marker = 1
    move = (2, 1)
    reward = 1
    temp_disc = 1

    reward_mods = agent._reward_move(state, marker, move, reward, temp_disc, game.ind_to_loc)

    assert isinstance(reward_mods, list)


def test_NeuralPlayer_process_state_reward(net):
    game = Game()
    lr = 0.25
    agent = NeuralPlayer(net, lr)
    state = np.reshape((0, 1, -1, 0, 1, 0, -1, 0, 0), game.board_shape)
    transform = {'func': None, 'args': {}}
    move = (2, 1)
    reward = 1
    temp_disc = 1
    equiv = False

    mod = agent._process_state_reward(state, transform, move, reward, temp_disc,
        equiv, game.ind_to_loc)
    
    assert isinstance(mod, ValueMod)


def test_NeuralPlayer_process_reward_no_reward(net):
    game = Game()
    lr = 0.25
    agent = NeuralPlayer(net, lr)
    marker = 1
    agent.buffer = [
        MoveRecord(state=np.reshape((0, -1, -1, 0, 0, 1, 0, 0, 1), game.board_shape),
                   move=(0, 0),
                   marker=marker),
        MoveRecord(state=np.reshape((1, -1, -1, 0, 0, 1, -1, 0, 1), game.board_shape),
                   move=(2, 1),
                   marker=marker)
    ]
    reward = 0
    expected_mods = []

    reward_mods = agent.process_reward(reward, game.ind_to_loc)

    assert reward_mods == expected_mods


def test_NeuralPlayer_process_reward_win(net):
    game = Game()
    lr = 0.25
    agent = NeuralPlayer(net, lr)
    marker = 1
    agent.buffer = [
        MoveRecord(state=np.reshape((0, -1, -1, 0, 0, 1, 0, 0, 1), game.board_shape),
                   move=(0, 0),
                   marker=marker),
        MoveRecord(state=np.reshape((1, -1, -1, 0, 0, 1, -1, 0, 1), game.board_shape),
                   move=(1, 1),
                   marker=marker)
    ]
    reward = 1

    agent.process_reward(reward, game.ind_to_loc)

    assert len(agent.reward_record) > 0
    # assert all([rm.target >= rm.previous for rm in agent.reward_record])
    # assert all([rm.result >= rm.previous for rm in agent.reward_record])


def test_NeuralPlayer_process_reward_lose(net):
    game = Game()
    lr = 0.25
    agent = NeuralPlayer(net, lr)
    marker = 1
    agent.buffer = [
        MoveRecord(state=np.reshape((0, -1, -1, 0, 0, 1, 0, 0, 1), game.board_shape),
                   move=(1, 1),
                   marker=marker)
    ]
    reward = -1

    reward_mods = agent.process_reward(reward, game.ind_to_loc)

    assert len(agent.reward_record) > 0
    # assert all([rm.target <= rm.previous for rm in agent.reward_record])
    # assert all([rm.result <= rm.previous for rm in agent.reward_record])
