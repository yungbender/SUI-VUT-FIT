import logging
import random
import pickle
from datetime import datetime
from configparser import ConfigParser

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import save_state, possible_attacks, probability_of_successful_attack, probability_of_holding_area
from dicewars.ai.xfrejl00.qtable import QTable
from dicewars.ai.xfrejl00.utils import *

class AlphaDice:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.players_order = players_order

        config = ConfigParser()
        config.read('dicewars/ai/xfrejl00/config.ini')
        self.update_qtable = config.getboolean('BASE', 'Train')
        self.snapshot_path = config['BASE']['SnapshotPath']
        self.moves_path = config['BASE']['SnapshotPath']  + "moves.pickle"
        self.learning_rate, self.epsilon, self.discount = load_parameters(self.snapshot_path)
        
        self.logger = logging.getLogger('AI')
        self.logger.info("Current time: " + datetime.now().strftime('%Y.%m.%d %H:%M:%S'))

        self.q_table = QTable(states_count=3, action_count=1, qvalue_check=True)

        if self.update_qtable:
            with open(self.moves_path , "wb") as f: # Create the empty file for saved moves
                pickle.dump([], f)
        
    def save_move_to_file(self, key):
        move_list = []
        with open(self.moves_path, 'rb') as f: # Load file
            move_list = pickle.load(f)
        
        move_list.append(key) # Add the current move
        with open(self.moves_path, 'wb') as f: # Save file
            pickle.dump(move_list, f)

    def get_qtable_key(self, board, source, target, action):
        """ State definition:
            - Probability of winning the encounter (very low, low, medium, high, very high)?)
            - Region size potential gain
            - Probability of keeping the area until the next turn (very low, low, medium, high, very high)
        """
        # Get the individual states
        success_probability = probability_of_successful_attack(board, source.get_name(), target.get_name())
        hold_probability = probability_of_holding_area(board, target.get_name(), source.get_dice() - 1, self.player_name)
        region_gain = region_size_potential_gain(board, source.get_name(), target, self.player_name)

        # Transform the probability into class probability (very low, low, medium, high, very high)
        success_probability = convert_probability_to_classes(success_probability)
        hold_probability = convert_probability_to_classes(hold_probability)

        return ((success_probability, hold_probability, region_gain), (action, ))

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        with open("xfrejl00.save", "wb") as f:
            save_state(f, board, self.player_name, self.players_order)
        attacks = list(possible_attacks(board, self.player_name))
        if not attacks: # No possible moves
            return EndTurnCommand()

        turn_source = None
        turn_target = None
        turn_key = None # Full Q-table entry for played move, so we can update its value after game
        turn_action = None
        self.q_table = self.q_table.load(self.snapshot_path + "snapshot.pickle")
        if random.uniform(0, 1) > self.epsilon or self.update_qtable == False: # Select action based on Q-table, don't play random when not training
            qvalue_max = float('-inf') # Default value is infinity because we want to always take the first possible move, no matter the Q-value
            for source, target in attacks:
                #print("\nPlayer dice: " + str(source.get_dice()) + ", area name: " + str(source.get_name()))
                #print("Enemy dice: " + str(target.get_dice()) + ", area name: " + str(target.get_name()))
                #print("Attack success probability: " + convert_probability_to_classes(success_probability))
                #print("Hold probability: " + convert_probability_to_classes(hold_probability))
                #print("Region size potential gain: " + str(region_gain))
                for action in ["attack", "defend"]:
                    key = self.get_qtable_key(board, source, target, action)
                    if key in self.q_table:
                        if self.q_table[key] > qvalue_max:
                            qvalue_max = self.q_table[key]
                            [turn_source, turn_target] = [source, target]
                            turn_key = key
                            turn_action = action
        else: # Select a random action
            turn_source, turn_target = random.choice(attacks)
            turn_action = random.choice(["attack", "defend"])

            if self.update_qtable:
                turn_key = self.get_qtable_key(board, turn_source, turn_target, turn_action)
                if turn_key not in self.q_table:
                    self.q_table[turn_key] = 0

        if self.update_qtable:
            #TODO: Update Q-table based on Bellman equation with immediate rewards (if self.update_qtable == True)
            self.q_table.save(self.snapshot_path + "snapshot.pickle")
            if turn_key is not None: # Only save the moves we were attacking in
                self.save_move_to_file(turn_key)

        if turn_source is None or turn_target is None or turn_action == "defend":
            return EndTurnCommand()
        else:
            return BattleCommand(turn_source.get_name(), turn_target.get_name())
