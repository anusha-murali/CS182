# -*- coding: utf-8 -*-
"""
CS 182 Problem Set 2: Python Coding Questions - Fall 2023
Due October 18, 2023 at 11:59pm
"""

### Package Imports ###
import abc
import random
from typing import Tuple, List, Type, Optional
### Package Imports ###

#### Coding Problem Set General Instructions - PLEASE READ ####
# 1. All code should be written in python 3.7 or higher to be compatible with the autograder
# 2. Your submission file must be named "pset2b.py" exactly
# 3. Submit this file to Pset2 - Coding Submission B
# 4. No additional outside packages can be referenced or called, they will result in an import error on the autograder
# 5. Function/method/class/attribute names should not be changed from the default starter code provided
# 6. All helper functions and other supporting code should be wholly contained in the default starter code declarations provided.
#    Functions and objects from your submission are imported in the autograder by name, unexpected functions will not be included in the import sequence


Action = str


class GameState(abc.ABC):
    """
    This is the abstract class for a game state. Your code should interact with the Ghost game through this interface.
    """

    @abc.abstractmethod
    def is_terminal(self) -> bool:
        """
        Method that returns a boolean value that indicates whether or not the state is a terminal state
        """

    @abc.abstractmethod
    def get_actions(self) -> List[Tuple[Action]]:
        """
        Method that returns a list of children of the current state
        """

    @abc.abstractmethod
    def generate_successor(self, action) -> "GameState":
        """
        Given an action, this method will return the game state that results from taking this action.
        """

    @abc.abstractmethod
    def value(self) -> float:
        """
        Returns the value of the current state if the state is a terminal state
        """


class GhostDictionary:
    """
    DO NOT MODIFY THIS CLASS!
    """
    def __init__(self, dictionary_file):
        self.characters = "abcdefghijklmnopqrstuvwxyz'"
        with open(dictionary_file, "r") as file:
            self.english_words_set = set(file.read().splitlines())

        # create set of all prefixes
        self.all_prefixes = set()
        for word in self.english_words_set:
            prefixes = self.find_prefixes(word)
            for pref in prefixes:
                self.all_prefixes.add(pref)

    # given word, return all of its prefixes
    @staticmethod
    def find_prefixes(word):
        return [word[:i] for i in range(0, len(word) + 1)]


class GhostGameState(GameState):
    """
    Each state holds three pieces of information
    - the current prefix in the game
    - the dictionary being used
    - the index of which player's turn it is

    DO NOT MODIFY THIS CLASS!
    """
    # this variables keeps track of the number of calls to `generate_successor`.
    generate_successor_counter = 0

    def __init__(self, prefix: str, ghost_dictionary: GhostDictionary, index=0):
        self.prefix = prefix
        self.dictionary = ghost_dictionary
        self.index = index

    def get_actions(self) -> List[Action]:
        if self.is_terminal():
            raise Exception("Cannot get the actions from a terminal state")
        legal_actions = []
        for letter in self.dictionary.characters:
            if self.prefix + letter in self.dictionary.all_prefixes:
                legal_actions.append(letter)
        return legal_actions

    def generate_successor(self, action: Action) -> "GameState":
        assert isinstance(action, Action), f"Your action {action} is not valid!"
        assert isinstance(self.prefix, str)
        GhostGameState.generate_successor_counter += 1
        return GhostGameState(self.prefix + action, self.dictionary, (self.index + 1) % 2)

    def is_terminal(self) -> bool:
        return self.prefix in self.dictionary.english_words_set

    def value(self) -> float:
        if not self.is_terminal():
            raise Exception("Not a terminal node")
        return ((-1) ** self.index) / len(self.prefix)


class MultiAgentSearchAgent(abc.ABC):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxAgent, AlphaBetaAgent.

      In the two-player game, each agent is indexed with either 0 or 1.
      The 0 player wants in the final value of the game being as large (positive) as possible.
      The 1 player wants the final value of the game being as small (negative) as possible.
    """

    def __init__(self, index):
        self.index = index

    @abc.abstractmethod
    def get_action(self, game_state: GameState) -> Action:
        """
        Returns the minimax action from the current game_state
        It may be helpful to use the functions `max_val` and `min_val` that you will fill in below,
        """


class MinimaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state: GameState) -> Action:
        if (game_state.index % 2 == 0):
            curVal, action = self.max_val(game_state)
        else:
            curVal, action = self.min_val(game_state)
        return action

    def max_val(self, game_state: GameState) -> Tuple[float, Optional[Action]]:
        """
        Given a `GameState` object, this function should return a tuple that contains
         - the maximum value that this agent is able to guarantee against any opponent
         - the action necessary to take from the current state corresponding to the maximum value that is returned
            - if `game_state` is already a terminal state, then this should be `None`.
        """
        if game_state.is_terminal():
            return (game_state.value(), None)
        else:
            value = -1000
            move = None
            curActions = game_state.get_actions()
            for action in curActions:
                (value2, action2) = \
                         self.min_val(game_state.generate_successor(action))
                if (value2 > value):
                    (value, move) = (value2, action)
            return (value, move)

    def min_val(self, game_state: GameState) -> Tuple[float, Optional[Action]]:
        """
        Given a `GameState` object, this function should return a tuple that contains
         - the minimum value that this agent is able to guarantee against any opponent
         - the action necessary to take from the current state corresponding to the minimum value that is returned
            - if `game_state` is already a terminal state, then this should be `None`.
        """
        if game_state.is_terminal():
            return (game_state.value(), None)
        else:
            value = 1000
            move = None
            curActions = game_state.get_actions()
            for action in curActions:
                (value2, action2) = \
                         self.max_val(game_state.generate_successor(action))
                if (value2 < value):
                    (value, move) = (value2, action)
            return (value, move)


class AlphaBetaAgent(MultiAgentSearchAgent):
    def get_action(self, game_state: GameState):
        if (game_state.index % 2 == 0):
            curVal, action = self.max_val(game_state, -1000, 1000)
        else:
            curVal, action = self.min_val(game_state, -1000, 1000)
        return action

    def max_val(self, game_state: GameState, alpha: float, beta: float) -> Tuple[float, Optional[Action]]:
        if game_state.is_terminal():
            return (game_state.value(), None)
        else:
            value = -1000
            move = None
            curActions = game_state.get_actions()
            for action in curActions:
                (value2, action2) = \
                         self.min_val(game_state.generate_successor(action), alpha, beta)
                if (value2 > value):
                    (value, move) = (value2, action)
                    alpha = max(alpha, value)
                if (value >= beta) or (alpha == beta):
                    return (value, move)
            return (value, move)

    def min_val(self, game_state: GameState, alpha: float, beta: float) -> Tuple[float, Optional[Action]]:
        if game_state.is_terminal() or (alpha == beta):
            return (game_state.value(), None)
        else:               
            value = 1000
            move = None
            curActions = game_state.get_actions()
            for action in curActions:
                (value2, action2) = \
                         self.max_val(game_state.generate_successor(action), alpha, beta)
                if (value2 < value):
                    (value, move) = (value2, action)
                    beta = min(beta, value)
                if (value <= alpha) or (alpha == beta):
                    return (value, move)
            return (value, move)


class RandomAgent(MultiAgentSearchAgent):
    def get_action(self, game_state: GameState) -> Action:
        return random.choice(game_state.get_actions())


class OptimizedAgainstRandomAgent(MultiAgentSearchAgent):
    """
    Implement the behavior of an agent that is optimized against a random agent here.
    Hint: it may be useful to implement helper functions like `min_val` and `max_val` just as you have done for
    the MinimaxAgent and the AlphaBetaAgent. You might also find it helpful to implement a third helper function
    that returns the expected value of a state from the RandomAgent's point of view.
    """
    def get_action(self, game_state: GameState) -> Action:
        # If the agent's index is 0, then she is a maximizer. If her index is 1, then
        # she is a minimizer
        # The other agent is the RandomAgent
        if (game_state.index % 2 == 0):
            curVal, action = self.max_val(game_state)
        else:
            curVal, action = self.min_val(game_state)
        return action

    def max_val(self, game_state: GameState) -> Tuple[float, Optional[Action]]:
        """
        Given a `GameState` object, this function should return a tuple that contains
         - the maximum value that this agent is able to guarantee against any opponent
         - the action necessary to take from the current state corresponding to the maximum value that is returned
         - if `game_state` is already a terminal state, then this should be `None`.
        """
        if game_state.is_terminal():
            return (game_state.value(), None)
        else:
            value = -1000
            move = None
            curActions = game_state.get_actions()
            utility = []
            for action in curActions:
                value2 = self.expected_val(game_state.generate_successor(action))
                utility.append((action,value2))
                if (value2 > value):
                    (value, move) = (value2, action)
            return (value, move)

    def min_val(self, game_state: GameState) -> Tuple[float, Optional[Action]]:
        """
        Given a `GameState` object, this function should return a tuple that contains
         - the minimum value that this agent is able to guarantee against any opponent
         - the action necessary to take from the current state corresponding to the minimum value that is returned
            - if `game_state` is already a terminal state, then this should be `None`.
        """
        if game_state.is_terminal():
            return (game_state.value(), None)
        else:
            value = 1000
            move = None
            curActions = game_state.get_actions()
            for action in curActions:
                value2 = self.expected_val(game_state.generate_successor(action))
                if (value2 < value):
                    (value, move) = (value2, action)
            return (value, move)


    def expected_val(self, game_state: GameState) -> float:
        """
        Given a `GameState` object, this function should return a tuple that contains
        - the expected value that this agent is able to generate
        - the action necessary to take from the current state corresponding to the random choice
        - if `game_state` is already a terminal state, then this should be `None`
        """
        if game_state.is_terminal():
            return game_state.value()
        else:
            maximize = True
            curActions = game_state.get_actions()
            # We assume if the agent's index == 0, then it is a maximizer and
            # if the agent's index == 1, then it is a minimizer
            # This means, when maximizer calls expected_val (which is a chance node)
            # the current node's index %2 = 1. Similarly when minimizer calls
            # expected_val, the current node's index %2 = 0
            if (game_state.index % 2 == 0):
                maximize = False
            totalVal = 0.0
            for action in curActions:
                if maximize:
                    (value, action2) = \
                        self.max_val(game_state.generate_successor(action))
                else:
                    (value, action2) = \
                        self.min_val(game_state.generate_successor(action))
                totalVal = totalVal + value
        return totalVal/len(curActions)


def play_game(
        dictionary: GhostDictionary,
        starting_prefix: str,
        starting_player: int,
        agent_class_0: Type[MultiAgentSearchAgent],
        agent_class_1: Type[MultiAgentSearchAgent],
        verbose: bool = True
) -> float:
    """
    This is a helper function for you to simulate the games locally.

    Given a GhostDictionary, a starting prefix, and the index of the starting player, and the class of the agent,
    this function simulates the full gameplay and returns the terminal state's value.
    """
    GhostGameState.generate_successor_counter = 0   # reset the counter for every game!
    start_state = GhostGameState(starting_prefix, dictionary, starting_player)
    if verbose:
        print(f"Starting prefix: {starting_prefix}. Starting agent: {starting_player}")
    assert not start_state.is_terminal(), "Prefix input is already a word! This is invalid."
    assert starting_prefix in dictionary.all_prefixes, "Prefix input is not in the set of valid prefixes!"
    state = start_state
    agents = [agent_class_0(0), agent_class_1(1)]
    current_index = starting_player
    while True:
        action = agents[current_index].get_action(state)
        state = state.generate_successor(action)
        if verbose:
            print(f"Agent {current_index} placed a {action}, bringing the current prefix to {state.prefix}")
        if state.is_terminal():
            if verbose:
                print(f"Total number of calls to `generate_successor`: {state.generate_successor_counter}")
                print("The game is over! Value: ", state.value())
            return state.value()
        current_index = (current_index + 1) % 2


def simulate_versus_random(dictionary: GhostDictionary, prefix: str, k: int = 10000) -> Tuple[float, float]:
    """This is a helper function for you to answer part (5)."""
    optimal_vs_random_value = 0
    minimax_vs_random_value = 0
    for _ in range(k):
        optimal_vs_random_value += play_game(dictionary, prefix, 0, OptimizedAgainstRandomAgent, RandomAgent, False)
        minimax_vs_random_value += play_game(dictionary, prefix, 0, MinimaxAgent, RandomAgent, False)
    optimal_vs_random_value /= k
    minimax_vs_random_value /= k
    return optimal_vs_random_value, minimax_vs_random_value


if __name__ == "__main__":
    dictionary = GhostDictionary("dictionary.txt")
    prefix = "wal"
    play_game(dictionary, prefix, 0, AlphaBetaAgent, AlphaBetaAgent)
