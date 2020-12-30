import logging
import random
import pickle
import numpy as np
import copy
import collections
from datetime import datetime
from configparser import ConfigParser

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import save_state, possible_attacks, probability_of_successful_attack, probability_of_holding_area
from dicewars.ai.xfrejl00.utils import region_size_potential_gain, convert_probability_to_classes


class QTableExc(Exception):
    pass


class QTablePickle(collections.UserDict):
    def __init__(self, states_count=None, action_count=None, qvalue_check=False):
        self.states_count = states_count
        self.action_count = action_count
        self.qvalue_check = qvalue_check

        super().__init__(self)

    def __getitem__(self, key: tuple):
        if not isinstance(key, tuple):
            raise QTableExc("Given key is not type tuple.")
        elif len(key) != 2:
            raise QTableExc("Accepting only keys in value ((state), (action))")

        # Key[0] is state, Key[1] is action
        if self.states_count is not None and len(key[0]) != self.states_count:
            raise QTableExc("Given state in key does not have given number of states.")

        if self.action_count is not None and len(key[1]) != self.action_count:
            raise QTableExc("Given action in key does not have given number of states.")

        return super().__getitem__(key)

    @staticmethod
    def load(where: str):
        with open(where, "rb") as file:
            return pickle.load(file)

class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.players_order = players_order

        self.snapshot_path = "dicewars/ai/alphadice-1.pickle"
        
        self.logger = logging.getLogger('AI')
        self.logger.info("Current time: " + datetime.now().strftime('%Y.%m.%d %H:%M:%S'))

        self.q_table = QTablePickle(states_count=3, action_count=1, qvalue_check=True)
        self.q_table = self.q_table.load(self.snapshot_path)
        self.rounds_without_move = 0

    def get_qtable_key(self, board, source, target, action):
        # Get the individual states
        success_probability = probability_of_successful_attack(board, source.get_name(), target.get_name())
        hold_probability = probability_of_holding_area(board, target.get_name(), source.get_dice() - 1, self.player_name)
        region_gain = region_size_potential_gain(board, source.get_name(), target, self.player_name)

        # Transform the probability into class probability (very low, low, medium, high, very high)
        success_probability = convert_probability_to_classes(success_probability)
        hold_probability = convert_probability_to_classes(hold_probability)
 
        return ((success_probability, hold_probability, region_gain), (action, ))

    def get_qtable_best_move(self, board, attacks):
        turn_source = None
        turn_target = None
        turn_key = None
        turn_action = None
        qvalue_max = float('-inf') # Default value is infinity because we want to always take the first possible move, no matter the Q-value
        
        # If we've played more than enough rounds without attack, we force attack, to not lose game by not submitting moves for 8 rounds
        if self.rounds_without_move < 7:
            actions = ["attack", "defend"]
        else:
            actions = ["attack"]
        
        for source, target in attacks:
            for action in actions:
                key = self.get_qtable_key(board, source, target, action)
                if key in self.q_table:
                    if self.q_table[key] > qvalue_max:
                        qvalue_max = self.q_table[key]
                        [turn_source, turn_target] = [source, target]
                        turn_key = key
                        turn_action = action
        return turn_source, turn_target, turn_key, turn_action

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        turn_key = None
        attacks = list(possible_attacks(board, self.player_name))
        if attacks:
            turn_source, turn_target, turn_key, turn_action = self.get_qtable_best_move(board, attacks)

        if not attacks or turn_action == "defend" or not turn_source or not turn_target: # Source or target can be null when there are missing records in Q-table
            self.rounds_without_move += 1
            return EndTurnCommand()
        else:
            self.rounds_without_move = 0
            return BattleCommand(turn_source.get_name(), turn_target.get_name())
