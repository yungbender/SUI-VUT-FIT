import logging
import random
import pickle
import numpy as np
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
        self.moves_in_turn = [] # List of all keys for moves performed during current turn, used when giving immediate rewards
        self.region_size_on_start = 0 # Biggest region size on turn start
        self.area_count_on_start = 0 # Area count on turn start

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

    def simulate_enemy_turns(self, board, players):
        for ai in players: 
            self.player_name = ai
            turn = None
            while not isinstance(turn, EndTurnCommand): # While the AI wants to play
                turn = self.ai_turn(board, 0, 0, 0) # We don't use the other 3 params
                if isinstance(turn, BattleCommand):
                    board = simulate_attack(board, turn)

        return board

    def simulate_game(self, board):
        name_backup = self.player_name
        self.update_qtable = False
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

        return board, new_dice

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        #with open("xfrejl00.save", "wb") as f:
        #    save_state(f, board, self.player_name, self.players_order)

        attacks = list(possible_attacks(board, self.player_name))
        if attacks:
            self.q_table = self.q_table.load(self.snapshot_path + "snapshot.pickle")
            if random.uniform(0, 1) > self.epsilon or self.update_qtable == False: # Select action based on Q-table, don't play random when not training
                turn_source, turn_target, turn_key, turn_action = self.get_qtable_best_move(board, attacks)
            else: # Select a random action
                turn_action = "attack"
                turn_source, turn_target = random.choice(attacks)
                if random.uniform(0,1) < 1 / (len(attacks) + 1): # Chance to defend = 1 / (number of possible attacks + 1) 
                    turn_action = "defend"
                
                if self.update_qtable:
                    turn_key = self.get_qtable_key(board, turn_source, turn_target, turn_action)
                    if turn_key not in self.q_table:
                        self.q_table[turn_key] = 0

            if self.update_qtable:
                self.q_table.save(self.snapshot_path + "snapshot.pickle")
                if turn_key is not None: # Only save the moves we were attacking in
                    self.save_move_to_file(turn_key)

        if not attacks or turn_action == "defend" or not turn_source or not turn_target: # Source or target can be null when there are missing records in Q-table
            if self.update_qtable and self.moves_in_turn: # Don't update Qtable unless we did at least one move
                # Before ending our turn, we simulate other players' turns and get Q-value of best move from this simulated board
                new_board, new_dice = self.simulate_game(board)
                new_attacks = list(possible_attacks(new_board, self.player_name))
                new_region_size = len(max(new_board.get_players_regions(self.player_name), key=len))
                new_area_size = len(new_board.get_player_areas(self.player_name))
                best_move = self.get_qtable_best_move(new_board, new_attacks)[2]
                if best_move:
                    max_qvalue_next_move = self.q_table[best_move]
                else: # If there are no possible moves on next simulated turn, we will probably lose during after this turn
                    max_qvalue_next_move = 0

                # Calculate reward
                #print("Region: " + str(self.region_size_on_start) + " -> " + str(new_region_size))
                #print("Area: " + str(self.area_count_on_start) + " -> " + str(new_area_size))
                reward = (new_region_size - self.region_size_on_start) * 0.5
                reward += (new_area_size - self.area_count_on_start) * 0.25 # Region size is more important

                for move in self.moves_in_turn: # Update Q-table value for all played moves in turn
                    # Bellman equation for new Q-table value calculation
                    #print("Move: ")
                    #print(move)
                    #print("Reward size: " + str(reward))
                    #print("Previous move value: " + str(self.q_table[move]))
                    #print("Best new possible move: " + str(max_qvalue_next_move))
                    self.q_table[move] = self.q_table[move] + self.learning_rate * (reward + self.discount * (max_qvalue_next_move - self.q_table[move])) # TU JE PROBLÉM, U ZÁPORNÝCH ČÍSEL TO BLBNE
                    #print("New move value: " + str(self.q_table[move]))

                self.moves_in_turn = []
            return EndTurnCommand()
        else:
            if not self.moves_in_turn: # Save biggest region size and area count on the first move
                self.region_size_on_start = len(max(board.get_players_regions(self.player_name), key=len))
                self.area_count_on_start = len(board.get_player_areas(self.player_name)) 
            self.moves_in_turn.append(turn_key)
            return BattleCommand(turn_source.get_name(), turn_target.get_name())
