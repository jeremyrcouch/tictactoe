from collections import namedtuple
import numpy as np
from typing import List, Tuple, Union


MoveRecord = namedtuple('MoveRecord', ['state', 'move', 'marker'])


class Player:
    def __init__(self, value_map: dict):
        self.buffer = []
        self.alpha = 0.5
        self.explore = True  # False -> exploit

    def record_move(self, state: np.ndarray, move: Tuple[int], marker: int):
        record = MoveRecord(state=state, move=move, marker=marker)
        self.buffer.append(record)