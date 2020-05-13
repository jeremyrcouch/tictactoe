from collections import namedtuple
import numpy as np
from typing import List, Tuple, Union


MoveRecord = namedtuple('MoveRecord', ['state', 'move', 'marker'])


class Player:
    def __init__(self, value_map: dict):
        self.buffer = []
        self.alpha = 0.5
        self.explore = True  # False -> exploit
        self.accepting_rewards = True

    def record_move(self, state: np.ndarray, move: Tuple[int], marker: int):
        record = MoveRecord(state=state, move=move, marker=marker)
        self.buffer.append(record)


class Human(Player):
    def __init__(self):
        self.buffer = []
        self.accepting_rewards = False

    def play(self, marker: int, game) -> Tuple[int]:
        """Player's action during their turn.

        Args:
            marker: int, player's marker in this game
            game: instance of Game

        Returns:
            loc: tuple of int, action (board location)
        """

        print(game.state)
        print("player {}'s turn".format(marker))
        row = input('row: ')
        col = input('col: ')
        loc = (int(row), int(col))
        return loc