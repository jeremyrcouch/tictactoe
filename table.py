from collections import namedtuple
from copy import deepcopy
from itertools import product
import numpy as np
from typing import List, Tuple, Union

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


class TablePlayer(Player):    
    def __init__(self, value_map: dict):
        super().__init__(self)
        self.value_map = deepcopy(value_map)
        self._min_reward_delta = 1/64
        self._discount_perc_decrease = 0.5

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
        # not using a model b/c tic-tac-toe is solved (not the point)
        # (model: predict future states and rewards in order to plan)
        
        # (value: total amount of reward expected to be received in future)
        # "action choices are made based on value judgements"

        # given state, determine actions and corresponding_values
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

        if reward == 0:
            return None
        
        ### Cascading Rewards ###
        # this approach cascades a reward backwards through moves
        # less reward is given to earlier moves
        # you have to be mindful of the moves in the buffer so you only apply rewards to relevant moves
        # with only a non-zero reward at the end of the game, this behaves the same whether
        #     rewards given every move or per game
        discount = 1
        # terminal = 0 if reward < 0 else 1
        reward_mods = []
        for entry in self.buffer[::-1]:
            match_state, transform = state_lookup(entry.state, self.value_map)
            action_values = self.value_map[match_state][entry.marker]
            adj_values = reverse_transforms(action_values, transform, ind_to_loc)
            current = adj_values[entry.move]

            # Option A: percentage of delta with terminal value of direction
            # issue: bigger deltas for rewards in the direction of the terminal value further from
            # percent = abs(reward*discount*self.alpha)
            # delta = max(self._min_reward_delta, abs((terminal - current)*percent))
            # Option B: just add value
            delta = max(self._min_reward_delta, abs(reward*discount*self.alpha))

            updated = np.clip(current + delta*np.sign(reward), a_min=0, a_max=1)
            undo = transform
            undo['args'] = {k: -undo['args'][k] for k in undo['args']}
            adj_move = [k for k in reverse_transforms({entry.move: 0}, undo, ind_to_loc)][0]
            self.value_map[match_state][entry.marker][adj_move] = updated
            discount *= self._discount_perc_decrease
            mod = ValueMod(state=match_state, move=adj_move, previous=current, new=updated)
            reward_mods.append(mod)
        ###

        ### Bellman Equation ###
        # maximum future reward for this state is the current reward
        #     plus the maximum future reward of the next state
        # Q[s,a] = Q[s,a] + α*(r + γ*np.max(Q[s1,:]) - Q[s,a])
        # TODO: discount player parameter?
        # discount = 0.75
        # entry = self.buffer[-1]
        # match_state, transform = state_lookup(entry.state, self.value_map)
        # action_values = self.value_map[match_state][entry.marker]
        # adj_values = reverse_transforms(action_values, transform, ind_to_loc)
        # current = adj_values[entry.move]

        # new_state = np.copy(entry.state)
        # new_state[entry.move[0], entry.move[1]] = entry.marker
        # new_match_state, _ = state_lookup(new_state, self.value_map)
        # new_action_values = self.value_map[new_match_state][entry.marker]
        # if isinstance(new_action_values, dict):
        #     max_future = max([new_action_values[a] for a in new_action_values])
        # else:
        #     max_future = new_action_values

        # updated = np.clip(current + self.alpha*(reward + (discount*max_future - current)),
        #                   a_min=0, a_max=1)
        # undo = transform
        # undo['args'] = {k: -undo['args'][k] for k in undo['args']}
        # adj_move = [k for k in reverse_transforms({entry.move: 0}, undo, ind_to_loc)][0]
        # self.value_map[match_state][entry.marker][adj_move] = updated
        # mod = ValueMod(state=match_state, move=adj_move, previous=current, new=updated)
        # reward_mods = [mod]
        ###

        return reward_mods

# bellman vs cascade
# - cascade is player1: win: 31.2%, lose: 13.9%, tie: 54.8%
# - player1 wins MA got up to 0.3
# TODO: combine bellman + cascade?


if __name__ == '__main__':
    init_value_map = initialize_value_map(INITIAL_VALUE)    
    player1 = TablePlayer(init_value_map)
    player2 = TablePlayer(init_value_map)

    # we can quickly hit a terminal, but then lose a ton with one bad outcome
    # lowering alpha over time fights this
    ALPHA_DECREASE_RATE = 0.25
    ALPHA_DECREASE_GAMES = 10000

    # first round of training: vs other player who is being trained
    # idea: learn faster vs smarter opponent
    trains = []
    for i in range(30000):
        if (i+1)%ALPHA_DECREASE_GAMES == 0:
            player1.alpha *= ALPHA_DECREASE_RATE
            player2.alpha *= ALPHA_DECREASE_RATE
        game = Game()
        play_game(game, player1, player2)
        trains.append(game.won)

    # second round of training: vs random opponent
    # idea: see if training has plateaued
    # re-run until plateaued
    refines = []
    player1.alpha = 0.5
    player3 = TablePlayer(init_value_map)
    player3.accepting_rewards = False
    for i in range(30000):
        if (i+1)%ALPHA_DECREASE_GAMES == 0:
            player1.alpha *= ALPHA_DECREASE_RATE
        game = Game()
        play_game(game, player1, player3)
        refines.append(game.won)

    # third round of training: vs other player who is being trained
    # idea: see if more room to grow
    player1.alpha = 0.5
    player2.alpha = 0.5
    for i in range(30000):
        if (i+1)%ALPHA_DECREASE_GAMES == 0:
            player1.alpha *= ALPHA_DECREASE_RATE
            player2.alpha *= ALPHA_DECREASE_RATE
        game = Game()
        play_game(game, player1, player2)
        trains.append(game.won)

    tests = []
    player1.explore = False
    # player3 = TablePlayer(init_value_map)
    # player3.accepting_rewards = False
    for _ in range(10000):
        game = Game()
        play_game(game, player1, player2)
        tests.append(game.won)
    print(value_frequencies(tests))

    # save_map = format_value_map(player1.value_map, tuple_to_str)
    # with open('value_map.json', 'w') as fp:
    #     json.dump(save_map, fp)
