import numpy as np
from typing import List, Tuple, Union

from src.utils.helpers import Game


class Player:
    def __init__(self, value_map: dict):
        self.buffer = []
        self.alpha = 0.5
        self.explore = True  # False -> exploit

    def play(self, marker: int, game: Game) -> Tuple[int]:
        pass

    def record_move(self, state: np.ndarray, move: Tuple[int], marker: int):
        self.buffer.append((state, move, marker))

    def process_reward(self, reward: Union[int, float], ind_to_loc: List[Tuple]):
        pass