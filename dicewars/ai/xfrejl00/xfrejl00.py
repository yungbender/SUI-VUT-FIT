import logging
import random
import pickle
import numpy as np
import copy
import os
import signal
from datetime import datetime
from configparser import ConfigParser
from torch import load, tensor

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import save_state, possible_attacks, probability_of_successful_attack, probability_of_holding_area
from dicewars.ai.xfrejl00.qtable import QTable
from dicewars.ai.xfrejl00.utils import *
from dicewars.ai.xfrejl00.classifier import LogisticRegressionMultiFeature as Classifier
from dicewars.ai.xfrejl00.dqn import LogisticRegressionMultiFeature as DQNetwork

DROPOUT_RATE = 0 # How many dataset inputs will get dropped
NB_FEATURES_CLASSIFIER = 22 # Number of classifier features
NB_FEATURES_DQN = 11 # 6 states, board states, custom stats
USE_DQN = False

class AlphaDice:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.players_order = players_order

        config = ConfigParser()
        config.read('dicewars/ai/xfrejl00/config.ini')
        self.update_qtable = config.getboolean('BASE', 'Train')
        self.snapshot_path = config['BASE']['SnapshotPath']
        self.moves_path = config['BASE']['SnapshotPath'] + "moves.pickle"
        self.learning_rate, self.epsilon, self.discount = load_parameters(self.snapshot_path)
        self.stats = GameStats(self.player_name, self.players_order, self.snapshot_path)
        self.ongoing_simulation = False # True when simulating other players
        
        self.logger = logging.getLogger('AI')
        self.logger.info("Current time: " + datetime.now().strftime('%Y.%m.%d %H:%M:%S'))

        self.q_table = QTable(states_count=5, action_count=1, qvalue_check=True)
        self.classifier = Classifier(NB_FEATURES_CLASSIFIER)
        self.classifier.load_state_dict(load(self.snapshot_path + f"classifier_model_{NB_FEATURES_CLASSIFIER}.pt"))
        self.classifier.eval()

        if USE_DQN:
            self.dqn = DQNetwork(NB_FEATURES_DQN)
            self.dqn.load_state_dict(load(self.snapshot_path + f"dqn_model_{NB_FEATURES_DQN}.pt"))
            self.dqn.eval()

        if os.path.isfile(self.snapshot_path + "snapshot.pickle"): # Snapshot already exists
            self.q_table = self.q_table.load(self.snapshot_path + "snapshot.pickle")

        self.logger.info(f"Epsilon: {self.epsilon}, Learning rate: {self.learning_rate}, Discount: {self.discount}, Train? {self.update_qtable}")

        if self.update_qtable:
            with shelve.open(self.moves_path, "n") as f:
                f["moves"] = []

    def save_dqtable_key(self, key):
        with open(self.snapshot_path + "dqn/statistics_raw.txt", "a+") as data:
            data.write(" ".join(str(item) for item in key) + "\n")

    def get_dqtable_key(self, board, source, target, action, winrate_change=0, reward=0, save=False):
        qtable_key = self.get_qtable_key(board, source, target, action, classes=False)
        dqtable_key = [list(x) for x in qtable_key] # Convert tuple of tuples to list of lists
        dqtable_key = [y for x in dqtable_key for y in x] # Flatten the list

        # Add our additional DQN stats to state
        additional_stats = [self.stats.rounds_without_move, target.get_dice(), self.stats.nb_players, self.stats.dice_per_area]#, winrate_change, self.stats.biggest_region_size, self.stats.area_share, self.stats.dice_share] 
        
        dqtable_key = dqtable_key + additional_stats
        assert(len(dqtable_key) == NB_FEATURES_DQN) # Check if the stat count is correct 

        if save:
            self.save_dqtable_key(dqtable_key)
        
        return dqtable_key

    def get_qtable_key(self, board, source, target, action, classes=True):
        # Get the individual states and action
        success_probability = probability_of_successful_attack(board, source.get_name(), target.get_name())
        hold_probability = probability_of_holding_area(board, target.get_name(), source.get_dice() - 1, self.player_name)
        region_gain = region_size_potential_gain(board, source.get_name(), target, self.player_name)
        region_destroy = region_size_potential_destroy(board, source, target, self.player_name)
        neighbor_count = neighboring_field_count(board, target)
        regions_at_risk = region_size_put_at_risk(board, source, target, self.player_name)

        if classes: # Transform the probability into class probability (very low, low, medium, high, very high)
            success_probability = convert_probability_to_classes(success_probability)
            hold_probability = convert_probability_to_classes(hold_probability)
            region_gain = convert_region_difference_to_classes(region_gain)
            region_destroy = convert_region_difference_to_classes(region_destroy)
            neighbor_count = convert_neighbor_count_to_classes(neighbor_count)
            regions_at_risk = convert_region_difference_to_classes(regions_at_risk)
        else: # Convert action to number instead of string to match dtype of other stats
            if action == "attack":
                action = 1
            else:
                action = 0

        return ((success_probability, hold_probability, region_gain, region_destroy, neighbor_count, regions_at_risk), (action, ))

    def get_qtable_best_move(self, board, attacks, enemy_simulation):
        turn_source = None
        turn_target = None
        turn_key = None
        turn_action = None
        qvalue_max = float('-inf') # Default value is infinity because we want to always take the first possible move, no matter the Q-value
        if not enemy_simulation: # We don't need stats when simulating enemies
            self.stats.get_game_statistics(board) # Generate current statistics

        # If we've played more than enough rounds without attack, we force attack, to not lose game by not submitting moves for 8 rounds
        if (self.stats.rounds_without_move < 7 and self.stats.dice_per_area < 8) or enemy_simulation:
            actions = ["attack", "defend"]
        else:
            actions = ["attack"]

        for source, target in attacks:
            for action in actions:
                key = self.get_qtable_key(board, source, target, action)
                if USE_DQN or key in self.q_table:
                    if USE_DQN and not enemy_simulation: # Get Q-value either from neural network or Q-table
                        dql_key = self.get_dqtable_key(board, source, target, action)
                        qvalue = self.dqn(tensor(dql_key)).item()
                    else:
                        if key in self.q_table:
                            qvalue = self.q_table[key]
                        else:
                            qvalue = self.q_table[key] = 0

                    if qvalue > qvalue_max:
                        qvalue_max = qvalue
                        [turn_source, turn_target] = [source, target]
                        turn_key = key
                        if USE_DQN and turn_key not in self.q_table: # We'll be also initializing regular Q-table when using Deep Q-learning
                            self.q_table[turn_key] = 0
                        turn_action = action
        return turn_source, turn_target, turn_key, turn_action

    def simulate_enemy_turns(self, board, players):
        for ai in players: 
            self.player_name = ai
            turn = None
            while not isinstance(turn, EndTurnCommand): # While the AI wants to play
                turn = self.ai_turn(board, 0, 0, 0, enemy_simulation=True) # We don't use the other 3 params
                if isinstance(turn, BattleCommand):
                    board = simulate_attack(board, turn)

        return board

    def simulate_game(self, board):
        name_backup = self.player_name
        self.update_qtable = False
        self.ongoing_simulation = True
        ai_order = self.players_order.index(name_backup) # Position of our AI in player order list
        
        # Simulate all other players after our AI until round ends and dice are awarded
        self.simulate_enemy_turns(board, self.players_order[ai_order+1:]) 
        
        # Calculate how much dice our AI will get after round ends
        ai_regions = board.get_players_regions(name_backup)
        new_dice = len(max(ai_regions, key=len)) # Length of longest list in regions list
        give_new_dice(board, self.players_order) # Distribute new dice across the board, because the round ended

        # Simulate all other players after our AI but after the dice were awarded 
        self.simulate_enemy_turns(board, self.players_order[:ai_order]) 

        self.player_name = name_backup
        self.update_qtable = True
        self.ongoing_simulation = False

        return board, new_dice

    def ai_turn(self, board, nb_moves, nb_turns, time_left, enemy_simulation=False):
        random.seed(random.random()) # New seed for an entire turn
        
        turn_key = None
        attacks = list(possible_attacks(board, self.player_name))
        if attacks:
            if random.uniform(0, 1) > self.epsilon or self.update_qtable == False: # Select action based on Q-table, don't play random when not training
                turn_source, turn_target, turn_key, turn_action = self.get_qtable_best_move(board, attacks, enemy_simulation)
            else: # Select a random action
                turn_action = "attack"
                turn_source, turn_target = random.choice(attacks)
                dice_per_area = board.get_player_dice(self.player_name) / len(board.get_player_areas(self.player_name))
                if random.uniform(0,1) < 1 / (len(attacks) + 1) and self.stats.rounds_without_move < 7 and turn_source.get_dice() < 8 and dice_per_area < 8: # Chance to defend = 1 / (number of possible attacks + 1) 
                    turn_action = "defend"
                
                if self.update_qtable:
                    turn_key = self.get_qtable_key(board, turn_source, turn_target, turn_action)
                    if turn_key not in self.q_table:
                        self.q_table[turn_key] = 0

        if self.update_qtable and turn_key: # Don't update Qtable unless we did a move
            # Before ending our turn, we simulate other players' turns and get Q-value of best move from this simulated board
            new_board = copy.deepcopy(board) # We must copy it so we don't change the original board
            new_board_no_new_move = copy.deepcopy(board) # For simulation without the new move

            if turn_action == "attack": # If we're gonna attack, simulate the attack first
                new_board = simulate_attack(new_board, BattleCommand(turn_source.get_name(), turn_target.get_name()))
                statistics_after = self.stats.get_game_statistics(new_board)
            else: # Defending stops our turn
                statistics_after = self.stats.get_game_statistics(new_board, on_turn=False)
            statistics_before = self.stats.get_game_statistics(board)

            new_board, new_dice = self.simulate_game(new_board)
            new_attacks = list(possible_attacks(new_board, self.player_name))
            new_area_size = len(new_board.get_player_areas(self.player_name))
            new_hidden_regions = hidden_region_count(new_board, self.player_name)

            # Get win probabilities after round simulation
            statistics_after_simulation_with_move = self.stats.get_game_statistics(new_board, on_turn=True, save=False) # Simulation with new move was already done
            new_board_no_new_move, _ = self.simulate_game(new_board_no_new_move)
            statistics_after_simulation_no_move = self.stats.get_game_statistics(new_board_no_new_move, on_turn=True, save=False) # Simulation with new move

            # Calculate reward
            area_count = len(board.get_player_areas(self.player_name)) 
            region_size = len(max(board.get_players_regions(self.player_name), key=len))
            hidden_regions = hidden_region_count(board, self.player_name)
            #print("Region: " + str(region_size) + " -> " + str(new_dice))
            #print("Area: " + str(area_count) + " -> " + str(new_area_size))
            reward = (new_dice - region_size) * 0.25 # We compare dice count at round end to current biggest region size
            reward += (new_area_size - area_count) * 0.05 # Region size is more important
            reward += (new_hidden_regions - hidden_regions) * 0.25
            reward = calculate_risk_reward_bonus(turn_key, reward)

            # Bellman equation for new Q-table value calculation
            #print("Move: ")
            #print(turn_key)
            #print("Reward size: " + str(reward))
            #print("Previous move value: " + str(self.q_table[turn_key]))
            #print(f"Probability to win: before - {self.classifier(tensor(statistics_before)):0.3f}, after - {self.classifier(tensor(statistics_after)):0.3f}")

            # Change Q-value based on immediate winrate change effect
            probability_before = self.classifier(tensor(statistics_before)).item()
            probability_after = self.classifier(tensor(statistics_after)).item()
            approximated_next_turn_qvalue = self.q_table[turn_key] * (1 + probability_after - probability_before) # Multiply current Q-value based on probability difference
            
            # Change Q-value based on winrate difference after simulated round with or without new attack (we don't value this as much as immediate effect)
            probability_with_move = self.classifier(tensor(statistics_after_simulation_with_move)).item()
            probability_without_move = self.classifier(tensor(statistics_after_simulation_no_move)).item()
            approximated_simulated_turn_qvalue = self.q_table[turn_key] * (1 + 0.20 * (probability_with_move - probability_without_move))

            # Calculate the total new approximated Q-value as weighted average
            approximated_qvalue = approximated_next_turn_qvalue * 0.75 + approximated_simulated_turn_qvalue * 0.25
            
            # Save the moves for DQN dataset:
            if random.uniform(0,1) > DROPOUT_RATE:
                if turn_action == "attack": # Generate stats for current board
                    self.stats.get_game_statistics(board, on_turn=True)
                else:
                    self.stats.get_game_statistics(board, on_turn=False)

                winrate_change = (probability_after - probability_before) * 0.75 + (probability_with_move - probability_without_move) * 0.25
                self.get_dqtable_key(board, turn_source, turn_target, turn_action, save=True, winrate_change=winrate_change, reward=reward)

            self.q_table[turn_key] = self.q_table[turn_key] + self.learning_rate * (reward + self.discount * approximated_qvalue - self.q_table[turn_key])
            self.q_table = give_reward_to_better_turns(self.q_table, reward, self.learning_rate, turn_key, 2, ["very low", "low", "medium", "high"])
            #print("New move value: " + str(self.q_table[turn_key]))

            # Save the move to the list of played moves
            save_move_to_file(turn_key, self.moves_path)

        if not attacks or turn_action == "defend" or not turn_source or not turn_target: # Source or target can be null when there are missing records in Q-table
            # Save various game statistics which will be used for dataset creation
            if not self.ongoing_simulation: # Check if the AI is playing, so we dont change the value when simulating other players
                self.stats.rounds_without_move += 1
            if random.uniform(0,1) > DROPOUT_RATE and self.update_qtable and len(board.get_player_areas(self.player_name)) > 0: # If the player is alive
                statistics = self.stats.get_game_statistics(board, on_turn=False, save=True)
            return EndTurnCommand()
        else:
            if not self.ongoing_simulation: # Check if the AI is playing, so we dont change the value when simulating other players
                self.stats.rounds_without_move = 0
            if random.uniform(0,1) > DROPOUT_RATE and self.update_qtable and len(board.get_player_areas(self.player_name)) > 0: # If the player is alive
                statistics = self.stats.get_game_statistics(board, on_turn=True, save=True)
            return BattleCommand(turn_source.get_name(), turn_target.get_name())

    def save_training(self):
        self.q_table.save(self.snapshot_path + "snapshot.pickle")
        self.q_table.close()
