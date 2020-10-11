import logging
import random
from datetime import datetime
from configparser import ConfigParser

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import save_state, possible_attacks, probability_of_successful_attack, probability_of_holding_area
from dicewars.ai.xfrejl00.qtable import QTable
from dicewars.ai.xfrejl00.utils import *

class AlphaDice:
    def __init__(self, player_name, board, players_order, update_qtable=False):
        self.player_name = player_name
        self.players_order = players_order
        self.update_qtable = update_qtable

        config = ConfigParser()
        config.read('dicewars/ai/xfrejl00/config.ini')
        self.train = config.getboolean('BASE', 'Train')
        self.snapshot_path = config['BASE']['SnapshotPath'] + "snapshot.pickle"
        self.epsilon = float(config['BASE']['Epsilon'])
        self.learning_rate = float(config['BASE']['LearningRate']) 
        self.discount = float(config['BASE']['Discount'])

        self.logger = logging.getLogger('AI')
        self.logger.info("Current time: " + datetime.now().strftime('%Y.%m.%d %H:%M:%S'))

        self.q_table = QTable(states_count=3, action_count=1, qvalue_check=True)
        self.q_table.load(self.snapshot_path)

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
        #with open("xfrejl00.save", "wb") as f:
        #    save_state(f, board, self.player_name, self.players_order)

        attacks = list(possible_attacks(board, self.player_name))
        if not attacks: # No possible moves
            return EndTurnCommand()

        attack_source = None
        attack_target = None
        action = None
        if random.uniform(0, 1) > self.epsilon: # Select action based on Q-table
            qvalue_max = 0
            for source, target in attacks:
                #print("\nPlayer dice: " + str(source.get_dice()) + ", area name: " + str(source.get_name()))
                #print("Enemy dice: " + str(target.get_dice()) + ", area name: " + str(target.get_name()))
                #print("Attack success probability: " + convert_probability_to_classes(success_probability))
                #print("Hold probability: " + convert_probability_to_classes(hold_probability))
                #print("Region size potential gain: " + str(region_gain))
                for action in ["attack", "defend"]:
                    key = self.get_qtable_key(board, source, target, action)
                    if key in self.q_table:
                        if self.q_table[key] >= qvalue_max:
                            [attack_source, attack_target] = [source, target]
        else: # Select a random action
            attack_source, attack_target = random.choice(attacks)

            key = self.get_qtable_key(board, attack_source, attack_target, action)
            if key not in self.q_table:
                self.q_table[key] = 0

        if attack_source is None or attack_target is None or action == "defend":
            return EndTurnCommand()
        else:
            #TODO: Save the chosen move so rewards received at the end of the game can be added too (if self.update_qtable == True)
            #TODO: Update Q-table based on Bellman equation with immediate rewards (if self.update_qtable == True)
            return BattleCommand(attack_source.get_name(), attack_target.get_name())
