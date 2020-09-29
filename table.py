from collections import namedtuple
from copy import deepcopy
from itertools import product
import json
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from utils.helpers import (Game, play_game, moving_average, state_transforms,
     check_states, state_transforms, reverse_transforms, reverse_function, state_to_actions,
     tuple_to_str, str_to_tuple, value_frequencies, moving_value_frequencies,
     plot_outcome_frequencies)
from utils.players import Player, Human, MoveRecord

INITIAL_VALUE = 0.5


ValueMod = namedtuple('ValueMod', ['state', 'move', 'previous', 'new'])


def initialize_value_map(init_val: float) -> dict:
    """Initialize a value map.

    Args:
        init_val: float, initial value

    Returns:
        init_value_map: dict, value map
    """

    prod_combs = product(Game.valid_markers + [Game.empty_marker],
                         repeat=Game.board_shape[0]**2)
    valid_combs = [pc for pc in prod_combs if abs(sum(pc)) < 2]

    non_dupes = []
    for vc in valid_combs:
        swap = tuple([elem*-1 for elem in vc])
        if swap not in non_dupes:
            non_dupes.append(vc)

    combs = []
    for nd in non_dupes:
        c_box = np.reshape(nd, Game.board_shape)
        rot90 = np.rot90(c_box)
        if tuple(rot90.flatten()) in combs:
            continue
        rot180 = np.rot90(c_box, k=2)
        if tuple(rot180.flatten()) in combs:
            continue
        rot270 = np.rot90(c_box, k=3)
        if tuple(rot270.flatten()) in combs:
            continue
        lr = np.fliplr(c_box)
        if tuple(lr.flatten()) in combs:
            continue
        ud = np.flipud(c_box)
        if tuple(ud.flatten()) in combs:
            continue
        combs.append(nd)

    # can't have more than one valid won state
    states = []
    for c in combs:
        game = Game()
        game.state = np.reshape(c, Game.board_shape)
        try:
            game._update_won()
            states.append(c)
        except ValueError:
            pass

    init_value_map = {
        s: {
            m: {
                a: init_val for a in state_to_actions(s, Game.ind_to_loc, Game.empty_marker)
            } for m in [-1, 1]
        } for s in states
    }

    for s in init_value_map:
        game = Game()
        game.state = np.reshape(s, game.board_shape)
        game._update_won()
        for m in init_value_map[s]:
            # won state: no actions, just reward value
            if game.won in game.valid_markers:
                init_value_map[s][m] = 1 if m == game.won else 0
            # full board: no actions, just initial value
            elif len(init_value_map[s][m]) == 0:
                init_value_map[s][m] = INITIAL_VALUE
            # cannot be marker's turn: no actions
            # NOTE: I don't explicitly reverse transform a marker swap
            #       so can't assume markers will match
            # elif sum(s) == m:
            #     init_value_map[s][m] = {}

    return init_value_map


def format_value_map(value_map: dict, key_func):
    """Format value map for saving or use.
    To save:
        with open('value_map.json', 'w') as fp:
            json.dump(save_map, fp)
    To load:
        with open('value_map.json', 'r') as fp:
            load_map = json.load(fp)
    
    Args:
        value_map: dict, value map
        key_func: function to modify keys

    Returns:
        mod_map: dict, value map modified
    """

    mod_map = {}
    for s in value_map:
        mark_map = {}
        for m in value_map[s]:
            if isinstance(value_map[s][m], dict):
                action_map = {}
                for a in value_map[s][m]:
                    action_map[key_func(a)] = value_map[s][m][a]
            else:
                action_map = value_map[s][m]
            mark_map[int(m)] = action_map
        mod_map[key_func(s)] = mark_map
    return mod_map


def state_lookup(state: np.ndarray, value_map: dict) -> (dict, dict):
    """Finding matching state in value map.

    Args:
        state: array, game board state
        value_map: dict, value map

    Returns:
        s: dict, matching state
        adj: dict, transform to get to matching state
    """

    states, adjs = check_states(state)
    for i, s in enumerate(states):
        mark_map = value_map.get(s, None)
        if mark_map:
            return s, adjs[i]
    raise ValueError('No matching state found')


def collect_values(value_map: dict, include_terminal: bool = False) -> List[float]:
    """Collect value map values.

    Args:
        value_map: dict, value map
        include_terminal: bool, True to include terminal (full board and won states)

    Returns:
        values: list of float
    """

    values = []
    for s in value_map:
        for m in value_map[s]:
            if isinstance(value_map[s][m], dict):
                for a in value_map[s][m]:
                    values.append(value_map[s][m][a])
            elif include_terminal:
                values.append(value_map[s][m])
    return values


def value_map_distribution(value_map: dict, bounds: List[float] = None):
    """Percent of values that fall in ranges.
    
    Args:
        value_map: dict, value map
        bound: list of float, boundaries to count values within
        
    Returns:
        dist: dict, distribution values
    """

    if not bounds:
        bounds = list(np.linspace(0, 1, 11))
        bounds[-1] += 0.01
    values = collect_values(value_map)
    total = len(values)
    dist = {}
    for low, up in zip(bounds[:-1], bounds[1:]):
        count = len([v for v in values if ((v >= low) and (v < up))])
        dist[(low, up)] = count/total
    
    return dist


def show_move_values(player: Player, game: Game):
    """Show learned values for game state.

    Args:
        player: instance of Player class
        game: instance of Game class
    """

    match_state, transform = state_lookup(game.state, player.value_map)
    action_values = agent.value_map.get(match_state, None)[game.mark]
    adj_values = reverse_transforms(action_values, transform, game.ind_to_loc)

    _, ax = plt.subplots(figsize=(4.5, 4.5))
    _ = plt.plot([1, 1], [0, -3], 'k-', linewidth=4)
    _ = plt.plot([2, 2], [0, -3], 'k-', linewidth=4)
    _ = plt.plot([0, 3], [-1, -1], 'k-', linewidth=4)
    _ = plt.plot([0, 3], [-2, -2], 'k-', linewidth=4)
    for x, y in game.ind_to_loc:
        if game.state[x, y] != 0:
            mark = 'x' if game.state[x, y] == 1 else 'o'
            plt.text(x + 0.275, -y - 0.725, mark, size=60)
        else:
            # TODO: add round(x, 2)
            plt.text(x + 0.35, -y - 0.575, adj_values[x, y], size=15)
            square = patches.Rectangle((x, -y - 1), 1, 1, linewidth=0,
                                       edgecolor='none', facecolor='r',
                                       alpha=adj_values[x, y]*0.75)
            ax.add_patch(square)
    _ = ax.axis('off')


class TablePlayer(Player):    
    def __init__(self, value_map: dict):
        super().__init__(self)
        self.value_map = deepcopy(value_map)
        self.discount_rate = 0.9  # discount future rewards
        self.temporal_discount_rate = 0.8  # discount credit assigned to earlier moves

    def play(self, marker: int, game: Game) -> Tuple[int]:
        """Player's action during their turn.

        Args:
            marker: int, player's marker in this game
            game: instance of Game

        Returns:
            loc: tuple of int, action (board location)
        """

        loc = self._policy(marker, game)
        return loc

    def _policy(self, marker: int, game: Game) -> Tuple[int]:
        match_state, transform = state_lookup(game.state, self.value_map)
        action_values = self.value_map.get(match_state, None)[marker]
        adj_values = reverse_transforms(action_values, transform, game.ind_to_loc)
        actions = [a for a in adj_values]
        raw_values = [adj_values[a] for a in actions]
        if sum(raw_values) <= 0:
            values = [1/len(raw_values) for v in raw_values]
        else:
            values = [v/sum(raw_values) for v in raw_values]
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
        """Update value map given reward.

        Args:
            reward: int or float, reward value
            ind_to_loc: list of tuple, game state index to board location map

        Returns:
            reward_mods: list of ValueMod, modifications to value for each move
        """
        
        temporal_discount = 1
        reward_mods = []
        # if the reward is 0, no update
        #     UNLESS the temporal_discount_rate is 0
        #     in this case, we need to process 0 rewards to transfer learning to earlier states
        #     we could use eligibility traces to update each state on every move..
        #     ..but that is less intuitive and won't be implemented here
        # if there's a non-zero reward (win/loss), assign credit to all moves (temporal difference)
        #     less credit is given to earlier moves, according to the temporal_discount_rate
        if (reward == 0) and (self.temporal_discount_rate > 0):
            return []
        elif reward == 0:
            entries = [self.buffer[-1]]
        else:
            entries = self.buffer[::-1]

        for entry in entries:
            # find the current value of (state, marker, move) combo
            match_state, transform = state_lookup(entry.state, self.value_map)
            action_values = self.value_map[match_state][entry.marker]
            adj_values = reverse_transforms(action_values, transform, ind_to_loc)
            current = adj_values[entry.move]

            # find the maximum value in the state resulting from the current move
            new_state = np.copy(entry.state)
            new_state[entry.move[0], entry.move[1]] = entry.marker
            new_match_state, _ = state_lookup(new_state, self.value_map)
            new_action_values = self.value_map[new_match_state][entry.marker]
            if isinstance(new_action_values, dict):
                max_future = max([new_action_values[a] for a in new_action_values])
            else:
                max_future = new_action_values

            # use the Bellman equation to update the current value
            updated = np.clip(current + temporal_discount*self.learning_rate*(reward
                + (self.discount_rate*max_future - current)),
                a_min=0, a_max=1)

            # reverse the transform to find the proper move to update.. and apply it
            undo = transform
            undo['args'] = {k: -undo['args'][k] for k in undo['args']}
            adj_move = [k for k in reverse_transforms({entry.move: 0}, undo, ind_to_loc)][0]
            self.value_map[match_state][entry.marker][adj_move] = updated

            # update temporal discount and record modification to value map
            temporal_discount *= self.temporal_discount_rate
            mod = ValueMod(state=match_state, move=adj_move, previous=current, new=updated)
            reward_mods.append(mod)

        return reward_mods


if __name__ == '__main__':
    init_value_map = initialize_value_map(INITIAL_VALUE)    
    agent = TablePlayer(init_value_map)
    competitor = TablePlayer(init_value_map)

    # train against a player who is learning how to beat you
    trains = []
    for _ in range(100000):
        game = Game()
        play_game(game, agent, competitor)
        trains.append(game.won)
        
    trains_mv = moving_value_frequencies(trains)
    plot_outcome_frequencies(trains_mv,
                            order=[1, 0, -1],
                            labels=['Agent Wins', 'Tie', 'Competitor Wins'])
    
    # test against a random player to see how much we've learned
    agent.explore = False
    agent.learning_rate = 0
    rando = TablePlayer(init_value_map)
    rando.learning_rate = 0

    tests = []
    for _ in range(10000):
        game = Game()
        play_game(game, agent, rando)
        tests.append(game.won)

    tests_freq = value_frequencies(tests)
    print(tests_freq)
    
    # train against random player to explore new situations
    agent.explore = True
    agent.learning_rate = 0.8

    rand_trains = []
    for _ in range(100000):
        game = Game()
        play_game(game, agent, rando)
        rand_trains.append(game.won)

    rand_mv = moving_value_frequencies(rand_trains)
    plot_outcome_frequencies(rand_mv,
                            order=[1, 0, -1],
                            labels=['Agent Wins', 'Tie', 'Random Wins'])

    agent.explore = False
    agent.learning_rate = 0
    tests_2 = []
    for _ in range(10000):
        game = Game()
        play_game(game, agent, rando)
        tests_2.append(game.won)

    tests_freq_2 = value_frequencies(tests_2)
    print(tests_freq_2)

    # additional training against competitor
    agent.explore = True
    agent.learning_rate = 0.8

    for _ in range(50000):
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
    
    save_map = format_value_map(agent.value_map, tuple_to_str)
    with open('value_map.json', 'w') as fp:
        json.dump(save_map, fp)
