from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from utils.helpers import (Game, play_game, value_frequencies, moving_value_frequencies,
    plot_outcome_frequencies)
from utils.players import Player, Human, MoveRecord


# TODO: correct properties?
ValueMod = namedtuple('ValueMod', ['state', 'move', 'previous', 'new'])


class NeuralPlayer(Player):    
    def __init__(self):
        super().__init__(self)

    def play(self, marker: int, game: Game) -> Tuple[int]:
        """Player's action during their turn.

        Args:
            marker: int, player's marker in this game
            game: instance of Game

        Returns:
            loc: tuple of int, action (board location)
        """

        loc = self._policy(marker, game)
        return loc

    def _policy(self, marker: int, game: Game) -> Tuple[int]:
        pass

    def process_reward(self, reward: Union[int, float], ind_to_loc: List[Tuple]) -> List[ValueMod]:
        """Learn from reward.

        Args:
            reward: int or float, reward value
            ind_to_loc: list of tuple, game state index to board location map

        Returns:
            reward_mods: list of ValueMod, modifications to value for each move
        """
        
        pass


if __name__ == '__main__':
    agent = NeuralPlayer()
    competitor = NeuralPlayer()

    # train against a player who is learning how to beat you
    trains = []
    for _ in range(100000):
        game = Game()
        play_game(game, agent, competitor)
        trains.append(game.won)
        
    trains_mv = moving_value_frequencies(trains)
    plot_outcome_frequencies(trains_mv,
                            order=[1, 0, -1],
                            labels=['Agent Wins', 'Tie', 'Competitor Wins'])
    
    # test against a random player to see how much we've learned
    agent.explore = False
    agent.learning_rate = 0
    rando = NeuralPlayer()
    rando.learning_rate = 0

    tests = []
    for _ in range(10000):
        game = Game()
        play_game(game, agent, rando)
        tests.append(game.won)

    tests_freq = value_frequencies(tests)
    print(tests_freq)
    
    # train against random player to explore new situations
    agent.explore = True
    agent.learning_rate = 0.8

    rand_trains = []
    for _ in range(100000):
        game = Game()
        play_game(game, agent, rando)
        rand_trains.append(game.won)

    rand_mv = moving_value_frequencies(rand_trains)
    plot_outcome_frequencies(rand_mv,
                             order=[1, 0, -1],
                             labels=['Agent Wins', 'Tie', 'Random Wins'])

    agent.explore = False
    agent.learning_rate = 0
    tests_2 = []
    for _ in range(10000):
        game = Game()
        play_game(game, agent, rando)
        tests_2.append(game.won)

    tests_freq_2 = value_frequencies(tests_2)
    print(tests_freq_2)

    # additional training against competitor
    agent.explore = True
    agent.learning_rate = 0.8

    for _ in range(50000):
        game = Game()
        play_game(game, agent, competitor)
        trains.append(game.won)
        
    final_tests = []
    agent.explore = False
    agent.learning_rate = 0
    for _ in range(10000):
        game = Game()
        play_game(game, agent, rando)
        final_tests.append(game.won)
        
    final_freq = value_frequencies(final_tests)
    print(final_freq)
