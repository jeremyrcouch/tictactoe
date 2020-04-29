import numpy as np
from typing import List, Tuple


class Game:
    _board_shape = (3, 3)
    _empty_marker = 0
    _valid_markers = [-1, 1]

    def __init__(self):
        self._state = np.zeros(self._board_shape, dtype=np.int8)
        self.done = False
        self.won = self._empty_marker
        self.turn = 0
    
    @property
    def state():
        return self._state
    
    @state.setter
    def state(self, loc: Tuple[int], marker: int) -> bool:
    # def mark(self, loc: Tuple[int], marker: int) -> bool:
        if marker not in self._valid_markers:
            msg = ('{} not one of the valid markers ({})'
                   .format(marker, self._valid_markers))
            # raise ValueError(msg)
            print(msg)
            return False
        if (marker != self.turn) and (self.turn in self._valid_markers):
            msg = "It's {}'s turn".format(self.turn)
            # raise ValueError(msg)
            print(msg)
            return False
        if self._state[loc[0], loc[1]] != 0:
            msg = 'Spot is already marked'
            # raise ValueError(msg)
            print(msg)
            return False
        self._state[loc[0], loc[1]] = marker
        self.turn = int(marker*-1)
        self._update_done()
        return True

    def _update_done(self):
        filled_spaces = np.sum(self._state != self._empty_marker)
        full_board = filled_spaces == self._state.size
        self._update_won()
        if full_board or self.won:
            self.done = True
        
    def _update_won(self):
        for marker in self._valid_markers:
            # making an assumption about def of _valid_markers here
            win_sum = marker*self._board_shape[0]
            vert = any(np.sum(self._state, axis=0) == win_sum)
            horiz = any(np.sum(self._state, axis=1) == win_sum)
            diag1 = np.sum(self._state.diagonal()) == win_sum
            diag2 = np.sum(np.fliplr(self._state).diagonal()) == win_sum
            if any([vert, horiz, diag1, diag2]):
                self.won = marker
                break


class Player:
    
    def __init__(self, value_map: dict):
        self.value_map = value_map

    def play(self, marker: int, state: Tuple[int]) -> bool:
        # TODO: in this case, play == _policy, but may not always be the case
        loc = self._policy(marker, state)
        return loc

    def _policy(self, marker: int, state: Tuple[int]) -> Tuple[int]:
        # not using a model b/c tic-tac-toe is solved (not the point)
        # (model: predict future states and rewards in order to plan)
        
        # (value: total amount of reward expected to be received in future)
        # "action choices are made based on value judgements"

        # given state, determine actions and corresponding_values
        action_values, transform = state_to_action_values(marker, state, self.value_map)
        adj_values = reverse_transforms(action_values, transform)
        actions = [a for a in adj_values]
        values = [adj_values[a] for a in actions]
        # take action with probability proportional to value
        loc = np.random.choice(actions, p=values)
        return loc


def reverse_transforms(action_values: dict, transform: dict) -> dict:
    if transform['type'] == 'rotate':
        adj_values = {reverse_rotate(act, adj['val']): action_values[act] 
                        for act in action_values}
    elif transform['type'] == 'fliplr':
        adj_values = {reverse_flip(act, np.fliplr): action_values[act]
                        for act in action_values}
    elif transform['type'] == 'flipud':
        adj_values = {reverse_flip(act, np.flipud): action_values[act]
                        for act in action_values}
    return adj_values


# TODO: combine these two functions, passing in func and keyword args
def reverse_flip(loc: Tuple[int], flip_func) -> Tuple[int]:
    inds = np.reshape([i for i in range(len(IND_TO_LOC))], (3,3))
    flip = flip_func(inds)
    ind = IND_TO_LOC.index(loc)
    new_coords = np.where(flip == ind)
    new_loc = (new_coords[0][0], new_coords[1][0])
    return new_loc


def reverse_rotate(loc: Tuple[int], rots: int) -> Tuple[int]:
    inds = np.reshape([i for i in range(len(IND_TO_LOC))], (3,3))
    rot = np.rot90(inds, k=-rots)
    ind = IND_TO_LOC.index(loc)
    new_coords = np.where(rot == ind)
    new_loc = (new_coords[0][0], new_coords[1][0])
    return new_loc


def check_states(state: Tuple[int]) -> (List[Tuple], List[int]):
    states, transforms = state_transforms(state)
    swap = tuple([elem*-1 for elem in state])
    swap_states, swap_transforms = state_transforms(swap, val_adj=-1)
    states.extend(swap_states)
    transforms.extend(swap_transforms)
    return states, transforms


def state_transforms(state: Tuple[int], val_adj: int = 1) -> (List[Tuple], List[dict]):
    states = [state]
    transforms = [{'type': 'none', 'val': val_adj*1}]
    box = np.reshape(state, (3, 3))
    for k in [1, 2, 3]:
        rot = np.rot90(box, k=k)
        states.append(tuple(rot.flatten()))
        transforms.append({'type': 'rotate', 'val': val_adj*k})
    lr = np.fliplr(box)
    states.append(tuple(lr.flatten()))
    transforms.append({'type': 'fliplr', 'val': val_adj*1})
    ud = np.flipud(box)
    states.append(tuple(ud.flatten()))
    transforms.append({'type': 'flipud', 'val': val_adj*1})
    return states, transforms

    
def state_to_action_values(marker: int, state: Tuple[int], value_map: dict) -> (dict, dict):
    # need to lookup state and all rotations and flips until find something!
    # need to also return what transformation we found match in
    #   so we can translate to proper loc, etc.
    # a game state, relative to a player, is state + marker
    states, adjs = check_states(state)
    for i, s in states:
        mark_map = value_map.get(s, None)
        if mark_map:
            return mark_map[marker], adjs[i]
    raise ValueError('No matching state found')


# def array_in_list(arr, arr_list):
#    return next((True for elem in arr_list if np.array_equal(elem, arr)), False)

# TODO: save to file, or func to build, or or or
# QUESTION: TODO: do we just need to add another layer for marker:
# {states: {markers: {actions: value}}}
# (0, 1, -1, 0, -1, 0, 1, 0, 0): {
#     -1: {(0, 0): 0.5, (1, 0): 0.7, ...},
#      1: {(0, 0): 0.2, (1, 0): 0.7, ...}
# }}
# {states: {actions: value}}
# TODO: store as list so we can use .index() for lookup?
from itertools import product
prod_combs = product([-1, 0, 1], repeat=9)
all_combs = [pc for pc in prod_combs]
valid_combs = [c for c in all_combs if abs(sum(c)) < 2]

non_dupes = []
for c in valid_combs:
    swap = tuple([elem*-1 for elem in c])
    if swap not in non_dupes:
        non_dupes.append(c)

combs = []
for c in non_dupes:
    c_box = np.reshape(c, (3, 3))
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

# TODO: generate
IND_TO_LOC = [
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2),
    (2, 0), (2, 1), (2, 2)
]

# given state, generate available actions
def state_to_actions(state: Tuple) -> List[Tuple]:
    inds = [i for i, mark in enumerate(state) if mark == 0]
    # NOTE: since len(state) only == 9, little to no benefit from:
    # inds = list(np.where(np.array(state) == 0)[0])
    actions = [IND_TO_LOC[ind] for ind in inds]
    return actions

# simplest possible setup:
# - start with empty state -> action value map
# - when player wins, pass reward = 1
# - use to update value map, cascading back through actions
# - (therefore, have to track actions during game)
# - when player loses, pass reward = -1 and do the same

init_val = 0.5
init_value_map = {
    c: {
        m: {
            a: init_val for a in state_to_actions(c)
        } for m in [-1, 1]
    } for c in combs
}
# player = Player(init_value_map)
# game = Game()
