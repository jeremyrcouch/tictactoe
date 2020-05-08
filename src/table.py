from itertools import product
import numpy as np
from typing import List, Tuple, Union

from src.utils.helpers import (Game, play_game, moving_average, state_transforms,
     checkstates, state_transforms, reverse_transforms, reverse_function, state_to_actions,
     print_outcomes)
from src.utils.players import Player

INITIAL_VALUE = 0.5


def initialize_value_map(init_val: float) -> dict:
    prod_combs = product(Game.valid_markers + [Game.empty_marker],
                         repeat=Game.board_shape[0]**2)
    all_combs = [pc for pc in prod_combs]
    valid_combs = [c for c in all_combs if abs(sum(c)) < 2]

    non_dupes = []
    for c in valid_combs:
        swap = tuple([elem*-1 for elem in c])
        if swap not in non_dupes:
            non_dupes.append(c)

    combs = []
    for c in non_dupes:
        c_box = np.reshape(c, Game.board_shape)
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
        combs.append(c)

    init_value_map = {
        c: {
            m: {
                a: init_val for a in state_to_actions(c, Game.ind_to_loc, Game.empty_marker)
            } for m in [-1, 1]
        } for c in combs
    }

    return init_value_map


def state_lookup(state: np.ndarray, value_map: dict) -> (dict, dict):
    states, adjs = checkstates(state)
    for i, s in enumerate(states):
        mark_map = value_map.get(s, None)
        if mark_map:
            return s, adjs[i]
    raise ValueError('No matching state found')


class TablePlayer(Player):    
    def __init__(self, value_map: dict):
        super().__init__(self)
        self.value_map = value_map.copy()
        self._min_reward_delta = 1/128

    def play(self, marker: int, game: Game) -> Tuple[int]:
        # TODO: in this case, play == _policy, but may not always be the case
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

    def process_reward(self, reward: Union[int, float], ind_to_loc: List[Tuple]):
        if reward == 0:
            return None
        
        discount = 1
        terminal = 0 if reward < 0 else 1
        for entry in self.buffer[::-1]:
            state, move, marker = entry
            match_state, transform = state_lookup(state, self.value_map)
            action_values = self.value_map[match_state][marker]
            adj_values = reverse_transforms(action_values, transform, ind_to_loc)
            current = adj_values[move]
            # TODO: should this really be a percent?
            # if we're at 0.75 and win , smaller delta than if we lose
            # maybe just handle with alpha (start it lower, for one - 0.25?)
            percent = abs(reward*discount*self.alpha)
            delta = max(self._min_reward_delta, abs((terminal - current)*percent))
            updated = np.clip(current + delta*np.sign(reward), a_min=0, a_max=1)
            undo = transform
            undo['args'] = {k: -undo['args'][k] for k in undo['args']}
            adj_move = [k for k in reverse_transforms({move: 0}, undo, ind_to_loc)][0]
            self.value_map[match_state][marker][adj_move] = updated
            discount *= 0.5
        # NOT wiping buffer here in case we want to troubleshoot


init_value_map = initialize_value_map(INITIAL_VALUE)
player1 = TablePlayer(init_value_map)
player2 = TablePlayer(init_value_map)

# what about player1 and player2 processing rewards, then play against rando player
# will they learn faster/better against smart competition?

wins = []
# TODO: tweak alpha during training
# we can quickly hit a terminal, but then lose a ton with one bad outcome
# lowering alpha over time fights this
# player1.alpha = ?
player1.explore = True
for _ in range(20000):
    game = Game()
    play_game(game, player1, player2)
    player1.process_reward(game.won, game.ind_to_loc)
    player2.process_reward(-game.won, game.ind_to_loc)
    wins.append(game.won)

ewins = []
player1.explore = False
player3 = Player(init_value_map)
for _ in range(1000):
    game = Game()
    play_game(game, player1, player3)
    # player1.process_reward(game.won, game.ind_to_loc)
    ewins.append(game.won)

# from matplotlib import pyplot as plt
# plt.plot(moving_average(ewins, n=1000))

player1.explore = False
exploit_wins = []
for _ in range(1000):
    game = Game()
    play_game(game, player1, player2)
    # player1.process_reward(game.won, game.ind_to_loc)
    exploit_wins.append(game.won)
# plt.plot(moving_average(exploit_wins, n=100))
# print(np.mean(exploit_wins))

# TODO: unit tests for funcs
# TODO: docstrings
# TODO: check type hints
# TODO: reward with each move
# TODO: when to update alpha? lower over time?
