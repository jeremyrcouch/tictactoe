from collections import namedtuple
import numpy as np
from typing import List, Tuple, Union

from utils.helpers import Game


MoveRecord = namedtuple('MoveRecord', ['state', 'move', 'marker'])


class Player:
    def __init__(self, value_map: dict):
        self.buffer = []
        self.alpha = 0.5
        self.explore = True  # False -> exploit

    def record_move(self, state: np.ndarray, move: Tuple[int], marker: int):
        record = MoveRecord(state=state, move=move, marker=marker)
        self.buffer.append(record)


class Human(Player):
    def __init__(self):
        self.buffer = []

    def play(self, marker: int, game: Game) -> Tuple[int]:
        """Player's action during their turn.

        Args:
            marker: int, player's marker in this game
            game: instance of Game

        Returns:
            loc: tuple of int, action (board location)
        """

        # row = input('row: ')
        # col = input('col: ')
        # loc = (int(row), int(col))
        print(game.state)
        loc = input('player {}, enter (row, col): '.format(marker))
        return loc