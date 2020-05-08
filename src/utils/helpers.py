import numpy as np
from typing import List, Tuple

from src.utils.players import Player


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


# not currently needed, keeping for reference
def array_in_list(arr, arr_list):
   return next((True for elem in arr_list if np.array_equal(elem, arr)), False)


def moving_average(vals, n=100):
    if n < 1:
        raise ValueError('n must be > 0')
    cum_vals = np.cumsum(vals)
    cum_vals[n:] = cum_vals[n:] - cum_vals[:-n]
    return cum_vals[n - 1:] / n


# side effects - ok?
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


def state_to_actions(state: Tuple[int], ind_to_loc: List[Tuple], empty: str) -> List[Tuple]:
    inds = [i for i, mark in enumerate(state) if mark == empty]
    actions = [ind_to_loc[ind] for ind in inds]
    return actions


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


def print_outcomes(wins: list):
    total = len(wins)
    print('win: {:.1%}, lose: {:.1%}, tie: {:.1%}'.format(
        len([w for w in wins if w == 1])/total,
        len([w for w in wins if w == -1])/total,
        len([w for w in wins if w == 0])/total
    ))