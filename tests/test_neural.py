import numpy as np
import pytest

import torch
import torch.nn as nn

from neural import (ValueMod, linear_net, one_hot_state, NeuralPlayer)
from utils.helpers import Game, state_to_actions
from utils.players import MoveRecord


DATA_PATH = 'tests/data/'


def test_linear_net():
    # arrange
    # act
    net = linear_net(Game())

    # assert
    assert isinstance(net, nn.Sequential)


def test_one_hot_state():
    # arrange
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

    # act
    ohe = one_hot_state(state, marker_order)

    # assert
    assert ohe.size == expected_size
    assert (ohe == expected_ohe).all()


def test_NeuralPlayer_state_values():
    # arrange
    game = Game()
    agent = NeuralPlayer(linear_net(game))
    state = np.reshape((0, 1, -1, 0, 1, 0, -1, 0, 0), game.board_shape)
    expected_len = game.board_shape[0]*game.board_shape[1]

    # act
    values = agent._state_values(state)

    # assert
    assert isinstance(values, torch.Tensor)
    assert len(values) == expected_len


@pytest.mark.parametrize(
    "marker, expected",
    [
        pytest.param(1, [0, 1, -1, 0, 1, 0, -1, 0, 0], id="no-swap"),
        pytest.param(-1, [0, -1, 1, 0, -1, 0, 1, 0, 0], id="swap")
    ],
)
def test_NeuralPlayer_adjust_state_for_marker(marker, expected):
    # arrange
    game = Game()
    agent = NeuralPlayer(linear_net(game))
    state = np.reshape((0, 1, -1, 0, 1, 0, -1, 0, 0), game.board_shape)
    expected_mod = np.reshape(expected, game.board_shape)

    # act
    state_mod = agent._adjust_state_for_marker(state, marker)

    # assert
    assert (state_mod == expected_mod).all()


def test_NeuralPlayer_policy():
    # arrange
    game = Game()
    agent = NeuralPlayer(linear_net(game))
    marker = 1
    state = np.reshape((0, 1, -1, 0, 1, 0, -1, 0, 0), game.board_shape)
    game.state = state

    # act
    move_values = agent._policy(marker, game)

    # assert
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
def test_NeuralPlayer_update_value_with_reward(value, reward):
    # arrange
    game = Game()
    agent = NeuralPlayer(linear_net(game))
    lr = 0.25
    temp_disc = 0.5

    # act
    updated = agent._update_value_with_reward(value, reward, lr, temp_disc)

    # assert
    assert updated >= 0
    assert updated <= 1
    if reward == 0:
        assert updated == value
    elif reward > 0:
        assert updated > value
    elif reward < 0:
        assert updated < value


def test_NeuralPlayer_calc_target_values():
    # arrange
    game = Game()
    agent = NeuralPlayer(linear_net(game))
    n_vals = game.board_shape[0]*game.board_shape[1]
    rand_vals = list(np.random.rand(n_vals, 1))
    values = torch.tensor(rand_vals, dtype=float)
    move_ind = 5
    current = values[move_ind].item()
    updated = current*1.1

    # act
    targets = agent._calc_target_values(values, current, updated, move_ind)

    # assert
    assert targets[move_ind].item() == updated
    assert torch.sum(values).item() == torch.sum(targets).item()


def test_NeuralPlayer_play():
    # arrange
    game = Game()
    agent = NeuralPlayer(linear_net(game))
    marker = 1
    state = np.reshape((0, 1, -1, 0, 1, 0, -1, 0, 0), game.board_shape)
    game.state = state
    actions = state_to_actions(tuple(state.flatten()), game.ind_to_loc, game.empty_marker)

    # act
    loc = agent.play(marker, game)

    # assert
    assert isinstance(loc, tuple)
    assert loc in actions


def test_NeuralPlayer_process_reward_no_reward():
    # arrange
    game = Game()
    agent = NeuralPlayer(linear_net(game))
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

    # act
    reward_mods = agent.process_reward(reward, game.ind_to_loc)

    # assert
    assert reward_mods == expected_mods


def test_NeuralPlayer_process_reward_win():
    # arrange
    game = Game()
    agent = NeuralPlayer(linear_net(game))
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

    # act
    reward_mods = agent.process_reward(reward, game.ind_to_loc)

    # assert
    assert all([rm.target >= rm.previous for rm in reward_mods])
    assert all([rm.result >= rm.previous for rm in reward_mods])


def test_NeuralPlayer_process_reward_lose():
    # arrange
    game = Game()
    agent = NeuralPlayer(linear_net(game))
    marker = 1
    agent.buffer = [
        MoveRecord(state=np.reshape((0, -1, -1, 0, 0, 1, 0, 0, 1), game.board_shape),
                   move=(1, 1),
                   marker=marker)
    ]
    reward = -1

    # act
    reward_mods = agent.process_reward(reward, game.ind_to_loc)

    # assert
    assert all([rm.target <= rm.previous for rm in reward_mods])
    assert all([rm.result <= rm.previous for rm in reward_mods])
