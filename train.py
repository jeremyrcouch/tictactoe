import numpy as np
from typing import Tuple


class Game:
    def __init__(self):
        self._board_shape = (3, 3)
        self._empty_marker = 0
        self.state = np.zeros(self._board_shape, dtype=np.int8)
        self._valid_markers = [-1, 1]
        self.done = False
        self.won = self._empty_marker

    def mark(self, loc: Tuple[int], marker: int) -> bool:
        if marker not in self._valid_markers:
            raise ValueError('{} not one of the valid markers ({})'
                             .format(marker, self._valid_markers))
        if self.state[loc[0], loc[1]] != 0:
            return False
        self.state[loc[0], loc[1]] = marker
        self._update_done()
        return True

    def _update_done(self):
        filled_spaces = np.sum(self.state != self._empty_marker)
        full_board = filled_spaces == self.state.size
        self._update_won()
        if full_board or self.won:
            self.done = True
        
    def _update_won(self):
        for marker in self._valid_markers:
            # making an assumption about def of _valid_markers here
            win_sum = marker*self._board_shape[0]
            vert = any(np.sum(self.state, axis=0) == win_sum)
            horiz = any(np.sum(self.state, axis=1) == win_sum)
            diag1 = np.sum(self.state.diagonal()) == win_sum
            diag2 = np.sum(np.fliplr(self.state).diagonal()) == win_sum
            if any([vert, horiz, diag1, diag2]):
                self.won = marker
                break