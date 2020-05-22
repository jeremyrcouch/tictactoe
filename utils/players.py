from collections import namedtuple
import numpy as np
from typing import Tuple, Union


MoveRecord = namedtuple('MoveRecord', ['state', 'move', 'marker'])


class Player:
    def __init__(self, value_map: dict):
        self.buffer = []  # holds player's moves during game
        self.learning_rate = 0.8  # how quickly new rewards alter policy (0 -> no learning)
        self.explore = True  # False -> exploit

    def record_move(self, state: np.ndarray, move: Tuple[int], marker: int):
        record = MoveRecord(state=state, move=move, marker=marker)
        self.buffer.append(record)
    
    def process_reward(self, reward: Union[int, float]):
        pass


class Human(Player):
    def __init__(self):
        self.buffer = []

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