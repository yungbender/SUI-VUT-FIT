import logging
import random
import pickle
import numpy as np
import copy
import os
import signal
from datetime import datetime
from configparser import ConfigParser

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import save_state, possible_attacks, probability_of_successful_attack, probability_of_holding_area
from dicewars.ai.xfrejl00.qtable import QTable
from dicewars.ai.xfrejl00.utils import *
from dicewars.ai.xfrejl00.classifier import LogisticRegressionMultiFeature

DROPOUT_RATE = 0.9 # How many dataset inputs will get dropped
NB_FEATURES = 11 # Number of classifier features

class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.players_order = players_order
        
        self.logger = logging.getLogger('AI')
        self.logger.info("Current time: " + datetime.now().strftime('%Y.%m.%d %H:%M:%S'))

        self.q_table = QTable(states_count=5, action_count=1, qvalue_check=True)
        self.q_table = self.q_table.load("dicewars/ai/alphadice-2.pickle")

    def get_qtable_key(self, board, source, target, action):
        # Get the individual states
        success_probability = probability_of_successful_attack(board, source.get_name(), target.get_name())
        hold_probability = probability_of_holding_area(board, target.get_name(), source.get_dice() - 1, self.player_name)
        region_gain = region_size_potential_gain(board, source.get_name(), target, self.player_name)
        region_destroy = region_size_potential_destroy(board, source, target, self.player_name)
        neighbor_count = neighboring_field_count(board, target)
        
        # Transform the probability into class probability (very low, low, medium, high, very high)
        success_probability = convert_probability_to_classes(success_probability)
        hold_probability = convert_probability_to_classes(hold_probability)
        region_gain = convert_region_difference_to_classes(region_gain)
        region_destroy = convert_region_difference_to_classes(region_destroy)
        neighbor_count = convert_neighbor_count_to_classes(neighbor_count)
 
        return ((success_probability, hold_probability, region_gain, region_destroy, neighbor_count), (action, ))

    def get_qtable_best_move(self, board, attacks):
        turn_source = None
        turn_target = None
        turn_key = None
        turn_action = None
        qvalue_max = float('-inf') # Default value is infinity because we want to always take the first possible move, no matter the Q-value
        for source, target in attacks:
            for action in ["attack", "defend"]:
                key = self.get_qtable_key(board, source, target, action)
                if key in self.q_table:
                    if self.q_table[key] > qvalue_max:
                        qvalue_max = self.q_table[key]
                        [turn_source, turn_target] = [source, target]
                        turn_key = key
                        turn_action = action
        return turn_source, turn_target, turn_key, turn_action


    def ai_turn(self, board, nb_moves, nb_turns, time_left):
        random.seed(random.random()) # New seed for an entire turn
        
        turn_key = None
        attacks = list(possible_attacks(board, self.player_name))
        if attacks:
            turn_source, turn_target, turn_key, turn_action = self.get_qtable_best_move(board, attacks)

        if not attacks or turn_action == "defend" or not turn_source or not turn_target: # Source or target can be null when there are missing records in Q-table
            return EndTurnCommand()
        else:
            return BattleCommand(turn_source.get_name(), turn_target.get_name())

    def save_training(self):
        self.q_table.close()
