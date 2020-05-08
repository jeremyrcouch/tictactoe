import numpy as np
from typing import List, Tuple, Union


class Player:
    def __init__(self, value_map: dict):
        self.buffer = []
        self.alpha = 0.5
        self.explore = True  # False -> exploit

    def record_move(self, state: np.ndarray, move: Tuple[int], marker: int):
        self.buffer.append((state, move, marker))