from itertools import product
import numpy as np
from typing import List, Tuple, Union


def state_to_actions(state: Tuple[int], ind_to_loc: List[Tuple]) -> List[Tuple]:
    inds = [i for i, mark in enumerate(state) if mark == 0]
    # NOTE: since len(state) only == 9, little to no benefit from:
    # inds = list(np.where(np.array(state) == 0)[0])
    actions = [ind_to_loc[ind] for ind in inds]
    return actions

# NOTE: not currently needed, keeping for reference
# def array_in_list(arr, arr_list):
#    return next((True for elem in arr_list if np.array_equal(elem, arr)), False)


def reverse_transforms(action_values: dict, transform: dict, ind_to_loc: List[Tuple]) -> dict:
    adj_values = {
        reverse_function(act, ind_to_loc, transform['func'], transform['args']): action_values[act] 
            for act in action_values
    }
    return adj_values


def reverse_function(loc: Tuple[int], ind_to_loc: List[Tuple],
                     func, func_args: dict = {}) -> Tuple[int]:
    if func is None:
        return loc
    inds = np.reshape([i for i in range(len(ind_to_loc))], (3, 3))
    transformed = func(inds, **func_args)
    ind = ind_to_loc.index(loc)
    coords = np.where(transformed == ind)
    new_loc = (coords[0][0], coords[1][0])
    return new_loc


def checkstates(state: np.ndarray) -> (List[Tuple], List[int]):
    flat = tuple(state.flatten())
    states, transforms = state_transforms(flat)
    swap = tuple([elem*-1 for elem in flat])
    swapstates, swap_transforms = state_transforms(swap)
    states.extend(swapstates)
    transforms.extend(swap_transforms)
    return states, transforms


def state_transforms(state: Tuple[int]) -> (List[Tuple], List[dict]):
    states = [state]
    transforms = [{'func': None, 'args': {}}]
    box = np.reshape(state, (3, 3))
    for k in [1, 2, 3]:
        rot = np.rot90(box, k=k)
        states.append(tuple(rot.flatten()))
        transforms.append({'func': np.rot90, 'args': {'k': -k}})
    lr = np.fliplr(box)
    states.append(tuple(lr.flatten()))
    transforms.append({'func': np.fliplr, 'args': {}})
    ud = np.flipud(box)
    states.append(tuple(ud.flatten()))
    transforms.append({'func': np.flipud, 'args': {}})
    return states, transforms

    
def state_to_action_values(marker: int, state: np.ndarray, value_map: dict) -> (dict, dict):
    states, adjs = checkstates(state)
    for i, s in enumerate(states):
        mark_map = value_map.get(s, None)
        if mark_map:
            return mark_map[marker], adjs[i]
    raise ValueError('No matching state found')


def state_lookup(state: np.ndarray, value_map: dict) -> (dict, dict):
    states, adjs = checkstates(state)
    for i, s in enumerate(states):
        mark_map = value_map.get(s, None)
        if mark_map:
            return s, adjs[i]
    raise ValueError('No matching state found')


# TODO: save to file, or func to build, or or or
def initialize_value_map() -> dict:
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

    init_val = 0.5
    init_value_map = {
        c: {
            m: {
                a: init_val for a in state_to_actions(c, Game.ind_to_loc)
            } for m in [-1, 1]
        } for c in combs
    }

    return init_value_map


def moving_average(vals, n=1000) :
    cum_vals = np.cumsum(vals)
    cum_vals[n:] = cum_vals[n:] - cum_vals[:-n]
    return cum_vals[n - 1:] / n


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

    def mark(self, loc: Tuple[int], marker: int) -> bool:
        if marker not in self.valid_markers:
            msg = ('{} not one of the valid markers ({})'
                   .format(marker, self.valid_markers))
            # raise ValueError(msg)
            print(msg)
            return False
        if (marker != self.turn) and (self.turn in self.valid_markers):
            msg = "It's {}'s turn".format(self.turn)
            # raise ValueError(msg)
            print(msg)
            return False
        if self.state[loc[0], loc[1]] != 0:
            msg = 'Spot is already marked'
            # raise ValueError(msg)
            print(msg)
            return False
        self.state[loc[0], loc[1]] = marker
        self.turn = int(marker*-1)
        self._update_done()
        return True

    def _update_done(self):
        filled_spaces = np.sum(self.state != self.empty_marker)
        full_board = filled_spaces == self.state.size
        self._update_won()
        if full_board or self.won:
            self.done = True
            self.turn = self.empty_marker
        
    def _update_won(self):
        for marker in self.valid_markers:
            # making an assumption about def of valid_markers here
            win_sum = marker*self.board_shape[0]
            vert = any(np.sum(self.state, axis=0) == win_sum)
            horiz = any(np.sum(self.state, axis=1) == win_sum)
            diag1 = np.sum(self.state.diagonal()) == win_sum
            diag2 = np.sum(np.fliplr(self.state).diagonal()) == win_sum
            if any([vert, horiz, diag1, diag2]):
                self.won = marker
                break


class Player:
    
    def __init__(self, value_map: dict):
        self.value_map = value_map.copy()
        self.buffer = []
        self.alpha = 0.5
        self.explore = True  # False -> exploit
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
        action_values, transform = state_to_action_values(marker, game.state, self.value_map)
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

    def record_move(self, state: np.ndarray, move: Tuple[int], marker: int):
        self.buffer.append((state, move, marker))

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


def play_game(game: Game, player1: Player, player2: Player, first: int = None):
    player1.buffer = []
    player2.buffer = []
    if first is not None:
        game.turn = game.valid_markers[first]
    else:
        game.turn = np.random.choice(game.valid_markers)

    while not game.done:
        prev_state = np.copy(game.state)
        prev_turn = game.turn
        # defining player1's marker as 1
        if game.turn == 1:
            cur_player = player1
        elif game.turn == -1:
            cur_player = player2
        else:
            break
        move = cur_player.play(game.turn, game)
        valid = game.mark(move, game.turn)
        if not valid:
            break
        cur_player.record_move(prev_state, move, prev_turn)


init_value_map = initialize_value_map()
player1 = Player(init_value_map)
player2 = Player(init_value_map)

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
def print_outcomes(wins: list):
    total = len(wins)
    print('win: {:.1%}, lose: {:.1%}, tie: {:.1%}'.format(
        len([w for w in wins if w == 1])/total,
        len([w for w in wins if w == -1])/total,
        len([w for w in wins if w == 0])/total
    ))

# TODO: unit tests for funcs
# TODO: docstrings
# TODO: check type hints
# TODO: reward with each move
# TODO: when to update alpha? lower over time?
