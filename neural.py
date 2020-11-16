from collections import namedtuple
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.helpers import (Game, play_game, value_frequencies, moving_value_frequencies,
    plot_outcome_frequencies, state_to_actions)
from utils.players import Player, Human, MoveRecord

device = torch.device('cpu')


# TODO: correct properties?
ValueMod = namedtuple('ValueMod', ['state', 'move', 'previous', 'target', 'result'])


def linear_net(game: Game, hidden_mult: int = 1, drop_prob: float = 0.1):
    """Linear network with a single hidden layer.
    
    Args:
        game: instance of Game class
        hidden_mult: int, hidden layer will have dim = hidden_mult * board_size
        drop_prob: float, dropout probability
    
    Returns:
        net: sequential neural network
    """
    
    n_spots = game.board_shape[0]*game.board_shape[1]
    n_inputs = (len(game.valid_markers) + 1)*n_spots
    n_hidden = int(n_inputs*hidden_mult)

    net = nn.Sequential(
        nn.Linear(n_inputs, n_hidden),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(n_hidden, n_spots),
        nn.Softmax(dim=0)
    )

    return net


def one_hot_state(state: np.ndarray, marker_order: List[int] = None) -> np.ndarray:
    """One-hot encode state of game.
    
    Args:
        state: array, game board state
        marker_order: list of int, order of markers for encoding
        
    Returns:
        ohe: array, encoded game board state
    """
    
    if marker_order is None:
        marker_order = [-1, 0, 1]
    flat = tuple(state.flatten())
    ohe = []
    for m in marker_order:
        ohe.extend([1 if s == m else 0 for s in flat])
        
    return np.array(ohe, dtype=np.int8)


def show_move_values(player: Player, game: Game):
    """Show learned values for game state.

    Args:
        player: instance of Player class
        game: instance of Game class
    """

    # TODO: function for this and play()
    actions = state_to_actions(tuple(game.state.flatten()), game.ind_to_loc, game.empty_marker)
    raw_values = player._policy(game.turn, game)
    valid_inds = [game.ind_to_loc.index(a) for a in actions]
    valid_values = [raw_values[ind] if ind in valid_inds else 0 for ind in range(len(raw_values))]
    if sum(valid_values) <= 0:
        values = [1/len(valid_values) for v in valid_values]
    else:
        values = [v/sum(valid_values) for v in valid_values]
    values = np.reshape(values, game.board_shape)

    _, ax = plt.subplots(figsize=(4.5, 4.5))
    _ = plt.plot([1, 1], [0, -3], 'k-', linewidth=4)
    _ = plt.plot([2, 2], [0, -3], 'k-', linewidth=4)
    _ = plt.plot([0, 3], [-1, -1], 'k-', linewidth=4)
    _ = plt.plot([0, 3], [-2, -2], 'k-', linewidth=4)
    for x, y in game.ind_to_loc:
        if game.state[x, y] != 0:
            mark = 'x' if game.state[x, y] == 1 else 'o'
            plt.text(y + 0.275, -x - 0.725, mark, size=60)
        else:
            plt.text(y + 0.35, -x - 0.575, round(values[x, y], 2), size=15)
            square = patches.Rectangle((y, -x - 1), 1, 1, linewidth=0,
                                       edgecolor='none', facecolor='r',
                                       alpha=values[x, y]*0.75)
            ax.add_patch(square)
    _ = ax.axis('off')


class NeuralPlayer(Player):    
    def __init__(self, net):
        super().__init__(self)
        self.learning_rate = 0.25
        self.temporal_discount_rate = 0.75  # discount credit assigned to earlier moves
        self.nn_learning_rate = 1e-2
        self.net = net
        self.opt = optim.Adam(self.net.parameters(), lr=self.nn_learning_rate)
        self.loss_fn = nn.MSELoss()

    def _state_values(self, state: np.ndarray) -> List[float]:
        hot = one_hot_state(state)
        x = torch.tensor(hot, dtype=float)
        values = self.net(x.float())
        return values
    
    def _adjust_state_for_marker(self, state: np.ndarray, marker: int) -> np.ndarray:
        state_mod = np.copy(state)
        if marker != 1:
            state_mod = state_mod*-1
        return state_mod
    
    def _policy(self, marker: int, game: Game) -> List[float]:
        # transform state so current player is always marker "1"
        # alternatively, could pass in current player's marker as a feature..
        # ..but that would be more difficult to troubleshoot
        state_mod = self._adjust_state_for_marker(game.state, marker)
        values = self._state_values(state_mod)
        return values.tolist()
    
    def _update_value_with_reward(self, value: float, reward: float, lr: float,
        temporal_discount: float) -> float:
        updated = np.clip(
            value + temporal_discount*lr*reward,
            a_min=0,
            a_max=1
        )
        return updated
    
    def _calc_target_values(self, values: List[float], current: float, updated: float,
        move_ind: int) -> List[float]:
        diff = updated - current
        target = values.detach().clone()
        target = target - (diff/(len(values) - 1))
        target[move_ind] = updated
        return target
    
    def play(self, marker: int, game: Game) -> Tuple[int]:
        """Player's action during their turn.

        Args:
            marker: int, player's marker in this game
            game: instance of Game

        Returns:
            loc: tuple of int, action (board location)
        """

        actions = state_to_actions(tuple(game.state.flatten()), game.ind_to_loc, game.empty_marker)
        raw_values = self._policy(marker, game)
        valid_inds = [game.ind_to_loc.index(a) for a in actions]
        valid_values = [raw_values[ind] for ind in valid_inds]
        if sum(valid_values) <= 0:
            values = [1/len(valid_values) for v in valid_values]
        else:
            values = [v/sum(valid_values) for v in valid_values]
        loc_inds = [i for i in range(len(values))]
        if self.explore:
            # take action with probability proportional to value
            loc_ind = np.random.choice(loc_inds, p=values)
        else:
            # exploit - take action with highest value
            loc_ind = loc_inds[np.argmax(values)]
        loc = actions[loc_ind]
        return loc

    def process_reward(self, reward: Union[int, float], ind_to_loc: List[Tuple]) -> List[ValueMod]:
        """Learn from reward.

        Args:
            reward: int or float, reward value
            ind_to_loc: list of tuple, game state index to board location map

        Returns:
            reward_mods: list of ValueMod, modifications to value for each move
        """
        
        temporal_discount = 1
        reward_mods = []
        # if the reward is 0, no update
        if reward == 0:
            return reward_mods
        
        # if there's a non-zero reward (win/loss), assign credit to all moves
        entries = self.buffer[::-1]
        for entry in entries:
            state_mod = self._adjust_state_for_marker(entry.state, entry.marker)
            values = self._state_values(state_mod)
            move_ind = ind_to_loc.index(entry.move)
            current = values[move_ind].item()

            # our network outputs probabilities for each action..
            # ..when we receive a reward for taking an action, we want to adjust that probability accordingly..
            # ..and adjust the other actions down to compensate (returning the sum to 1)
            # less credit is given to earlier moves, according to the temporal_discount_rate
            updated = self._update_value_with_reward(current, reward,
                self.learning_rate, temporal_discount)
            target = self._calc_target_values(values, current, updated, move_ind)
            loss = self.loss_fn(values, target)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            
            new_values = self._state_values(state_mod)
            result = new_values[move_ind].item()

            # update temporal discount and record modification to value map
            temporal_discount *= self.temporal_discount_rate
            mod = ValueMod(
                state=entry.state,
                move=entry.move,
                previous=current,
                target=updated,
                result=result
            )
            reward_mods.append(mod)

        return reward_mods


if __name__ == '__main__':
    agent = NeuralPlayer(linear_net(Game(), hidden_mult=2, drop_prob=0.1))
    competitor = NeuralPlayer(linear_net(Game(), hidden_mult=2, drop_prob=0.1))

    # train against a player who is learning how to beat you
    trains = []
    for _ in range(900000):
        game = Game()
        play_game(game, agent, competitor)
        trains.append(game.won)

    trains_mv = moving_value_frequencies(trains)
    plot_outcome_frequencies(
        trains_mv,
        order=[1, 0, -1],
        labels=['Agent Wins', 'Tie', 'Competitor Wins']
    )

    # test against a random player to see how much we've learned
    agent.explore = False
    agent.learning_rate = 0
    rando = NeuralPlayer(linear_net(Game()))
    rando.learning_rate = 0

    tests = []
    for _ in range(10000):
        game = Game()
        play_game(game, agent, rando)
        tests.append(game.won)

    tests_freq = value_frequencies(tests)
    print(tests_freq)
    
    agent.explore = True
    agent.learning_rate = 0.25

    rand_trains = []
    for _ in range(500000):
        game = Game()
        play_game(game, agent, rando)
        rand_trains.append(game.won)

    rand_mv = moving_value_frequencies(rand_trains)
    plot_outcome_frequencies(
        rand_mv,
        order=[1, 0, -1],
        labels=['Agent Wins', 'Tie', 'Random Wins']
    )

    agent.explore = False
    agent.learning_rate = 0

    tests_2 = []
    for _ in range(10000):
        game = Game()
        play_game(game, agent, rando)
        tests_2.append(game.won)

    tests_freq_2 = value_frequencies(tests_2)
    print(tests_freq_2)

    agent.explore = True
    agent.learning_rate = 0.25

    for _ in range(100000):
        game = Game()
        play_game(game, agent, competitor)
        trains.append(game.won)
        
    final_tests = []
    agent.explore = False
    agent.learning_rate = 0
    for _ in range(10000):
        game = Game()
        play_game(game, agent, rando)
        final_tests.append(game.won)
        
    final_freq = value_frequencies(final_tests)
    print(final_freq)

    game = Game()
    game.state = np.reshape((0, 1, -1, 0, 1, 0, -1, 0, 0), game.board_shape)
    game.mark = 1
    show_move_values(agent, game)