from copy import deepcopy
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils.players import Player, Human


class Game:
    board_shape = (3, 3)
    empty_marker = 0
    valid_markers = [-1, 1]
    ind_to_loc = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2)
    ]

    def __init__(self):
        self.state = np.zeros(self.board_shape, dtype=np.int8)
        self.done = False
        self.won = self.empty_marker
        self.turn = self.empty_marker
    
    @property
    def first(self):
        """Which player went first this game."""
        if self.turn == self.empty_marker:
            return None
        total = np.sum(self.state)
        if total != 0:
            return total
        return self.turn

    def determine_reward(self, marker: int) -> Union[int, float]:
        """Reward criteria for game.

        Args:
            marker: int, player's marker for this game
        
        Returns:
            int or float, reward
        """

        if self.won == marker:
            return 1
        elif self.won in self.valid_markers:
            return -1
        else:
            return 0
    
    def mark(self, loc: Tuple[int], marker: int) -> (bool, Union[int, float]):
        """Take a player's attempted move and, if valid, apply it to the game state.

        Args:
            loc: tuple of int, board location to mark
            marker: int, player's marker for this game

        Returns:
            bool, True if move was valid - False otherwise
            reward: int or float, reward given move
        """

        if marker not in self.valid_markers:
            msg = ('{} not one of the valid markers ({})'
                   .format(marker, self.valid_markers))
            print(msg)
            return False, 0
        if (marker != self.turn) and (self.turn in self.valid_markers):
            print("It's {}'s turn".format(self.turn))
            return False, 0
        if self.state[loc[0], loc[1]] != 0:
            print('Spot is already marked')
            return False, 0
        if self.done:
            print('Game is already over')
            return False, 0
        self.state[loc[0], loc[1]] = marker
        self.turn = int(marker*-1)
        self._update_done()
        reward = self.determine_reward(marker)
        return True, reward

    def _update_done(self):
        filled_spaces = np.sum(self.state != self.empty_marker)
        full_board = filled_spaces == self.state.size
        self._update_won()
        if full_board or self.won:
            self.done = True
        
    def _update_won(self):
        winners = []
        for marker in self.valid_markers:
            win_sum = marker*self.board_shape[0]
            vert = any(np.sum(self.state, axis=0) == win_sum)
            horiz = any(np.sum(self.state, axis=1) == win_sum)
            diag1 = np.sum(self.state.diagonal()) == win_sum
            diag2 = np.sum(np.fliplr(self.state).diagonal()) == win_sum
            if any([vert, horiz, diag1, diag2]):
                winners.append(marker)
        if len(winners) > 1:
            raise ValueError('More than 1 winner.')
        elif len(winners) == 1:
            self.won = winners[0]


def tuple_to_str(state: tuple) -> str:
    """(1, 0, -1) -> '10-1'"""
    return ''.join([str(s) for s in state])


def str_to_tuple(state_str: str) -> tuple:
    """'10-1' -> (1, 0, -1)"""
    temp = [s for s in state_str]
    state = []
    i = 0
    while i < len(temp):
        try:
            st = int(temp[i])
            i += 1
        except ValueError:
            st = int(''.join(temp[i:i+2]))
            i += 2
        state.append(st)
    return tuple(state)


# not currently needed, keeping for reference
def array_in_list(arr: np.ndarray, arr_list: List[np.ndarray]):
   return next((True for elem in arr_list if np.array_equal(elem, arr)), False)


def moving_average(vals: list, n: int = 100) -> np.ndarray:
    """Calculate moving average of values.

    Args:
        vals: list, values
        n: int, number of previous points to calculate average from

    Returns:
        array, moving average
    """

    if n < 1:
        raise ValueError('n must be > 0')
    cum_vals = np.cumsum(vals)
    cum_vals[n:] = cum_vals[n:] - cum_vals[:-n]
    return cum_vals[n - 1:] / n


# side effects
def play_game(game: Game, player1: Player, player2: Player, first: int = None, actions: list = None):
    """Play a single game.

    Args:
        game: instance of Game
        player1: instance of player class
        player2: instance of player class
        first: int, index of Game.valid_markers (0 or 1) to specify whose turn is first
        actions: list, predefined series of actions to take
    """

    player1.buffer = []
    player2.buffer = []
    if actions is not None:
        # TODO: don't ignore first arg here, check consistent or not defined
        if len(actions[0]) > len(actions[1]):
            first = 1
        elif len(actions[1]) > len(actions[0]):
            first = 0
        pmoves = [(m for m in p) for p in actions]
    else:
        pmoves = [(m for m in []) for p in range(2)]
    if first is not None:
        game.turn = game.valid_markers[first]
    else:
        game.turn = np.random.choice(game.valid_markers)

    marks = game.valid_markers[::-1]
    while not game.done:
        prev_state = np.copy(game.state)
        prev_turn = game.turn
        act_index = marks.index(game.turn)
        # defining player1's marker as 1
        cur_player = player1 if game.turn == 1 else player2
        try:
            move = next(pmoves[act_index])
        except StopIteration:
            move = cur_player.play(game.turn, game)
        valid, reward = game.mark(move, game.turn)
        if not valid and not isinstance(cur_player, Human):
            break
        cur_player.record_move(prev_state, move, prev_turn)
        if cur_player.learning_rate > 0:
            cur_player.process_reward(reward, game.ind_to_loc)
    
    # reward for player who did not make the last move (won/lost/tie)
    cur_player = player1 if game.turn == 1 else player2
    if cur_player.learning_rate > 0:
        reward = game.determine_reward(game.turn)
        cur_player.process_reward(reward, game.ind_to_loc)


# side effects
def play_round_of_games(player1: Player, player2: Player, n: int,
                        first: int = None, actions: list = None, silent: bool = False):
    outcomes = []
    for i in range(n):
        if (not silent) and ((i + 1) % int(n/5) == 0):
            print('{:.0%}'.format((i + 1)/n))
        game = Game()
        play_game(game, agent, competitor, first=first, actions=actions)
        outcomes.append(game.won)
        
    return outcomes


# when we lose, we want to train that game to get better at that series of situations
def replay_loss(agent: Player, player2: Player, game: Game, n: int):
    step = 0
    first = game.first
    p1buff = deepcopy(agent.buffer)
    p2buff = deepcopy(player2.buffer)
    while step < len(p1buff):
        actions = [
            [b.move for b in p1buff[:-step]],
            [b.move for b in p2buff[:-step]]
        ]
        outcomes = play_round_of_games(agent, player2, n, first=first,
                                       actions=actions, silent=True)
        step += 1
    agent.explore = False
    actions = [
        [b.move for b in p1buff[:-1]],
        [b.move for b in p2buff[:-1]]
    ]
    outcomes = play_round_of_games(agent, RandomPlayer(), n, first=first,
                                   actions=actions, silent=True)
    agent.explore = True
    return outcomes


def state_to_actions(state: Tuple[int], ind_to_loc: List[Tuple], empty: str) -> List[Tuple]:
    """Lookup the potential actions given the current state of the game.

    Args:
        state: tuple of int, game board state
        ind_to_loc: list of tuple, game state index to board location map
        empty: str, marker for empty board space

    Returns:
        actions: list of tuple, potential actions (board locations)
    """

    inds = [i for i, mark in enumerate(state) if mark == empty]
    actions = [ind_to_loc[ind] for ind in inds]
    return actions


def check_states(state: np.ndarray) -> (List[Tuple], List[dict]):
    """Find the list of states to check for a match and their transforms.

    Args:
        state: array, game board state

    Returns:
        states: list of tuple, states to check for a match
        transforms: list of dict, transforms to move from input state to states to check
    """

    flat = tuple(state.flatten())
    states, transforms = state_transforms(flat)
    swap = tuple([elem*-1 for elem in flat])
    swapstates, swap_transforms = state_transforms(swap)
    states.extend(swapstates)
    transforms.extend(swap_transforms)
    return states, transforms


def state_transforms(state: Tuple[int]) -> (List[Tuple], List[dict]):
    """Transform a state by rotation and symmetry.

    Args:
        state: tuple of int, game board state
    
    Returns:
        states: list of tuple, transformed states
        transforms: list of dict, transforms to move from input state to output states
    """

    states = [state]
    transforms = [{'func': None, 'args': {}}]
    box = np.reshape(state, (3, 3))
    for k in [1, 2, 3]:
        rot = np.rot90(box, k=k)
        states.append(tuple(rot.flatten()))
        transforms.append({'func': np.rot90, 'args': {'k': k}})
    lr = np.fliplr(box)
    states.append(tuple(lr.flatten()))
    transforms.append({'func': np.fliplr, 'args': {}})
    ud = np.flipud(box)
    states.append(tuple(ud.flatten()))
    transforms.append({'func': np.flipud, 'args': {}})
    return states, transforms


def reverse_transforms(action_values: dict, transform: dict, ind_to_loc: List[Tuple]) -> dict:
    """Map locations based on transform to matching state.

    Args:
        action_values: dict, action to value map
        transform: dict, transform function and args
        ind_to_loc: list of tuple, game state index to board location map

    Returns:
        adj_values: dict, action to value map with actions transformed to matching state
    """

    adj_values = {
        reverse_function(act, ind_to_loc, transform['func'], transform['args']): action_values[act] 
            for act in action_values
    }
    return adj_values


def reverse_function(loc: Tuple[int], ind_to_loc: List[Tuple],
                     func, func_args: dict = {}) -> Tuple[int]:
    """Find location from reversing transform.

    Args:
        loc: tuple of int, move location
        ind_to_loc: list of tuple, game state index to board location map
        func: transform function
        func_args: dict, transform function args to reverse it

    Returns:
        new_loc: tuple of int, location from reversing transform
    """

    if func is None:
        return loc
    inds = np.reshape([i for i in range(len(ind_to_loc))], (3, 3))
    transformed = func(inds, **func_args)
    ind = ind_to_loc.index(loc)
    coords = np.where(transformed == ind)
    new_loc = (coords[0][0], coords[1][0])
    return new_loc


def value_frequencies(values: list):
    """Frequency of values in list.

    Args:
        values: list
    
    Returns:
        freqs: dict, frequency of each value
    """

    vals = np.array(values)
    total = len(vals)
    freqs = {}
    for val in np.unique(vals):
        freqs[val] = len(vals[vals == val])/total
    return freqs


def moving_value_frequencies(values: list, n: int = 1000):
    """Calculate moving frequency of values.

    Args:
        values: list
        n: int, size of moving window (number of values)

    Returns:
        freqs: dict, moving frequencies
    """

    if n < 0:
        raise ValueError('n must be >= 0')

    unique_vals = set(values)
    freqs = {val: [] for val in unique_vals}
    for i in range(len(values) - n + 1):
        window_freqs = value_frequencies(values[i:i+n])
        for val in unique_vals:
            freqs[val].append(window_freqs.get(val, 0))

    return freqs


def plot_outcome_frequencies(freqs: dict, order: list = None, labels: list = None):
    """Show stacked plot of outcomes (player wins and ties).

    Args:
        freqs: dict, frequency of each outcome over games
        order: list, specified order to display stacks in
        labels: list, labels for outcomes
    """

    font_size = 16

    if order is None:
        if {1, 0, -1} == set([v for v in freqs]):
            order = [1, 0, -1]
        else:
            order = [v for v in freqs]
    else:
        for outcome in order:
            if outcome not in freqs:
                freqs[outcome] = [0]*len(freqs[[v for v in freqs][0]])

    if labels is None:
        if {1, 0, -1} == set([v for v in freqs]):
            labels = ['P1 Wins', 'Tie', 'P2 Wins']
        else:
            labels = [str(v) for v in freqs]
    
    y = [freqs[k] for k in order]
    _, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(range(len(y[0])), y, labels=labels, alpha=0.4)
    plt.legend(loc='upper right', framealpha=0.2, fontsize=font_size-4)
    plt.xlabel('Game #', fontsize=font_size)
    plt.ylabel('Frequency', fontsize=font_size)
    ax.tick_params(labelsize=font_size)
    plt.xlim([0, len(y[0])])
    plt.ylim([0, 1])