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
    plot_outcome_frequencies, state_to_actions, check_states, reverse_function)
from utils.players import Player, Human, MoveRecord

device = torch.device('cpu')  # if you are running on a machine with a nvidia gpu, use 'cuda' here


ValueMod = namedtuple('ValueMod', ['state', 'move', 'previous', 'target', 'result', 'equiv'])


def linear_net(game: Game, hidden_layers: list = None, drop_prob: float = 0.1):
    """Linear network with a flexible number of hidden layers.
    
    Args:
        game: instance of Game class
        hidden_layers: list of int, length is number of layers
            value means that layer will have that times many more features than the previous layer
        drop_prob: float, dropout probability
    
    Returns:
        net: sequential neural network
    """
    
    if hidden_layers is None:
        hidden_layers = [1]
    
    n_spots = game.board_shape[0]*game.board_shape[1]
    n_inputs = len(game.valid_markers)*n_spots
    prev_feats = n_inputs
    linear_layers = []
    for layer_mult in hidden_layers:
        cur_feats = int(prev_feats*layer_mult)
        linear_layers.append(nn.Linear(prev_feats, cur_feats))
        prev_feats = cur_feats

    net = nn.Sequential(
        *linear_layers,
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(prev_feats, n_spots),
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
        marker_order = [-1, 1]
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
    def __init__(self, net, lr: float, temp_rate: float = 0.8, nn_lr: float = 1e-3):
        super().__init__(self)
        self.learning_rate = lr
        self.temporal_discount_rate = temp_rate  # discount credit assigned to earlier moves
        self.nn_learning_rate = nn_lr
        self.net = net
        self.opt = optim.Adam(self.net.parameters(), lr=self.nn_learning_rate)
        self.loss_fn = nn.MSELoss()
        self.min_probability = 1/81
    
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
        values_raw = self._state_values(state_mod)
        values = values_raw.tolist()
        valid_values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        return list(valid_values)
    
    def _update_value_with_reward(self, value: float, reward: float, lr: float,
        temporal_discount: float) -> float:
        updated = np.clip(
            value + temporal_discount*lr*reward,
            a_min=0,
            a_max=1
        )
        return updated
    
    def _calc_target_values(self, values: List[float], current: float, updated: float,
        move_ind: int, valid_inds: list) -> List[float]:
        diff = updated - current
        target = values.detach().clone()

        # set non-valid indexes to 0 target value
        for i in [j for j in range(len(values)) if j not in valid_inds]:
            target[i] = 0

        # don't violate 0 to 1 range
        adj_inds = [j for j in valid_inds if j != move_ind]
        if len(adj_inds) > 0:
            sub = diff/len(adj_inds)
        else:
            sub = diff
        for i in adj_inds:
            target[i] = np.clip(target[i].detach() - sub, a_min=0, a_max=1)

        target[move_ind] = updated
        if sum(target) > 0:
            target = target/sum(target)
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
        if len(actions) == 0:
            raise Error('no available actions')
        raw_values = self._policy(marker, game)
        valid_inds = [game.ind_to_loc.index(a) for a in actions]
        valid_values = [raw_values[ind] for ind in valid_inds]
        if sum(valid_values) <= 0:
            values = [1/len(valid_values) for v in valid_values]
        else:
            values = [v/sum(valid_values) for v in valid_values]
        loc_inds = [i for i in range(len(values))]
        if self.explore:
            # limit the minimum probability for an action
            probs = [v if v > self.min_probability else self.min_probability for v in values]
            probs = [v/sum(probs) for v in probs]
            # take action with probability proportional to value
            loc_ind = np.random.choice(loc_inds, p=probs)
        else:
            # exploit - take action with highest value
            loc_ind = loc_inds[np.argmax(values)]
        loc = actions[loc_ind]
        return loc

    def _equivalent_states_to_reward(self, state: np.ndarray) -> (list, list):
        equiv_states, equiv_transforms = check_states(state)
        non_swap_ind = int(len(equiv_states)/2)
        equiv_no_swap = equiv_states[:non_swap_ind]
        equiv_no_swap = [np.reshape(s, (3, 3)) for s in equiv_no_swap]
        trans_no_swap = equiv_transforms[:non_swap_ind]
        equivs = []
        transforms = []
        for s, t in zip(equiv_no_swap, trans_no_swap):
            if not any([np.array_equal(s, es) for es in equivs]):
                equivs.append(s)
                transforms.append(t)
        return equivs, transforms

    def _reward_move(self, state: np.ndarray, marker: int, move: tuple, reward: float,
                     temp_disc: float, ind_to_loc: List[Tuple]) -> ValueMod:
        reward_mods = []
        adj_state = self._adjust_state_for_marker(state, marker)
        equiv_states, equiv_transforms = self._equivalent_states_to_reward(adj_state)
        for s, t in zip(equiv_states, equiv_transforms):
            equiv = not np.array_equal(s, adj_state)
            mod = self._process_state_reward(s, t, move, reward, temp_disc, equiv, ind_to_loc)
            reward_mods.append(mod)
        return reward_mods

    def _process_state_reward(self, state: np.ndarray, transform: dict, move: tuple, reward: float,
                              temp_disc: float, equiv: bool, ind_to_loc: List[Tuple]):
        actions = state_to_actions(tuple(state.flatten()), ind_to_loc, 0)
        valid_inds = [ind_to_loc.index(a) for a in actions]
        values = self._state_values(state)
        adj_move = reverse_function(move, ind_to_loc, transform['func'], transform['args'])
        move_ind = ind_to_loc.index(adj_move)
        current = values[move_ind].item()

        # our network outputs probabilities for each action..
        # ..when we receive a reward for taking an action, we want to adjust that probability accordingly..
        # ..and adjust the other actions down to compensate (returning the sum to 1)
        # less credit is given to earlier moves, according to the temporal_discount_rate
        updated = self._update_value_with_reward(current, reward,
            self.learning_rate, temp_disc)
        target = self._calc_target_values(values, current, updated, move_ind, valid_inds)
        loss = self.loss_fn(values, target)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        new_values = self._state_values(state)
        result = new_values[move_ind].item()
        mod = ValueMod(
            state=state,
            move=move,
            previous=current,
            target=updated,
            result=result,
            equiv=equiv
        )
        return mod
            
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
        if reward == 0:
            return reward_mods
        
        # if there's a non-zero reward (win/loss), assign credit to all moves
        entries = self.buffer[::-1]
        for entry in entries:
            mods = self._reward_move(entry.state, entry.marker, entry.move, reward,
                                     temporal_discount, ind_to_loc)
            reward_mods.extend(mods)
            temporal_discount *= self.temporal_discount_rate
        
        self.reward_record = reward_mods


if __name__ == '__main__':
    lr = 0.25
    nn_lr = 1e-3
    temp_rate = 0.8
    layers = [2, 1, 0.5] # [1]
    drop_prob = 0.0 # 0.05

    agent = NeuralPlayer(
        linear_net(Game(), hidden_layers=layers, drop_prob=drop_prob),
        lr=lr, temp_rate=temp_rate, nn_lr=nn_lr
    )
    competitor = NeuralPlayer(
        linear_net(Game(), hidden_layers=layers, drop_prob=drop_prob),
        lr=lr, temp_rate=temp_rate, nn_lr=nn_lr
    )
    rando = RandomPlayer()

    n = 50000
    outcomes = []
    seq_outcomes = {vm: {out: 0 for out in (Game.valid_markers + [Game.empty_marker])} for vm in Game.valid_markers}
    for i in range(n):
        if ((i + 1) % int(n/5)) == 0:
            print('{:.0%}'.format((i + 1)/n))
        game = Game()
        play_game(game, agent, competitor)
        outcomes.append(game.won)
        seq_outcomes[game.first][game.won] += 1

    n = 2000
    outcomes2 = []
    seq_outcomes2 = {vm: {out: 0 for out in (Game.valid_markers + [Game.empty_marker])} for vm in Game.valid_markers}
    for i in range(n):
        if ((i + 1) % int(n/5)) == 0:
            print('{:.0%}'.format((i + 1)/n))
        
        for first in [0, 1]:
            for start in Game.ind_to_loc:
                game = Game()
                if first == 0:
                    actions = [[start], []]
                else:
                    actions = [[], [start]]
                play_game(game, agent, competitor, first=first, actions=actions)
                outcomes2.append(game.won)
                seq_outcomes2[game.first][game.won] += 1

    mv2 = moving_value_frequencies(outcomes2)
    plot_outcome_frequencies(mv2, order=[1, 0, -1], labels=['Agent Wins', 'Tie', 'Competitor Wins'])


    agent.explore = False # True
    agent.learning_rate = 0 # lr
    test = play_round_of_games(agent, rando, 1000)
    freqs = value_frequencies(test)
    print(freqs)


    agent.explore = True
    agent.learning_rate = lr
    rand_outcomes = play_round_of_games(agent, rando, 100000)
    mv = moving_value_frequencies(rand_outcomes)
    plot_outcome_frequencies(mv, order=[1, 0, -1], labels=['Agent Wins', 'Tie', 'Other Wins'])


    game = Game()
    game.state = np.reshape((0, 0, 0, 0, 0, 0, 0, 0, 0), game.board_shape)
    # game.state = np.reshape((0, 1, -1, 0, 1, 0, -1, 0, 0), game.board_shape)
    # game.state = np.reshape((-1, 0, 0, 0, 1, 0, 0, 0, -1), game.board_shape)
    # game.state = np.reshape((0, 0, 0, 0, -1, 0, 0, 0, 0), game.board_shape)
    game.turn = 1
    show_move_values(agent, game)