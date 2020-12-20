from dicewars.client.game.board import Board
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import probability_of_successful_attack
from itertools import combinations
import numpy as np
import random
import shelve


class GameStats():
    def __init__(self, player_name, players_order, snapshot_path):
        self.player_name = player_name
        self.players_order = players_order
        self.snapshot_path = snapshot_path
        self.rounds_without_move = 0
        self.board_stats = []

    def get_game_statistics(self, board, on_turn=True, save=False):
        total_dice_count = 0
        for i in self.players_order:
            total_dice_count += board.get_player_dice(i)
        
        # Number of remaining players
        self.nb_players = board.nb_players_alive()

        # Biggest region size
        self.biggest_region_size = len(max(board.get_players_regions(self.player_name), key=len))

        # Hidden region count
        self.hidden_regions = hidden_region_count(board, self.player_name)

        # Total share of board owned
        self.area_share = len(board.get_player_areas(self.player_name)) / len(board.areas)

        # Total share of dice owned
        self.dice_share = board.get_player_dice(self.player_name) / total_dice_count

        # Total dice count
        self.dice_total = board.get_player_dice(self.player_name)

        # Region count
        self.region_count = len(board.get_players_regions(self.player_name))

        if len(board.get_player_areas(self.player_name)) > 0: # Player is alive
            # Average dice per area
            self.dice_per_area = board.get_player_dice(self.player_name) / len(board.get_player_areas(self.player_name))

            # Total share of areas on border
            self.areas_on_border = len(board.get_player_border(self.player_name)) / len(board.get_player_areas(self.player_name))

            # Total share of areas in danger
            self.areas_in_danger = len(get_areas_in_danger(board, self.player_name)) / len(board.get_player_areas(self.player_name)) 
        else:
            self.areas_in_danger = self.areas_on_border = self.dice_per_area = 0
        
        # Size of biggest region size of opponent
        self.most_regions_opponent = get_most_opponent_regions(board, self.player_name, self.players_order)

        # Average distance between player areas
        self.avg_region_distance = average_region_distance(board, self.player_name)

        # Number of players ahead of player
        if on_turn:
            self.players_ahead = 0
        else:
            self.players_ahead = board.nb_players_alive() - 1
        
        # How many players play after us before dice are distributed
        self.turns_until_dice = turns_until_end_of_round(self.player_name, self.players_order)

        # Get enemy state [region_size, area_count, dice_count] of individual enemies, sorted by region_size
        self.enemy_state = get_enemy_state(board, self.player_name, self.players_order)

        if save:
            with open(self.snapshot_path + "classifier/statistics.txt", "a+") as f:
                f.write(f"{self.nb_players} {self.biggest_region_size} {self.hidden_regions} {self.area_share} {self.dice_share} {self.dice_total} {self.region_count} {self.dice_per_area} {self.areas_on_border} {self.areas_in_danger} {self.avg_region_distance} {self.players_ahead} {self.turns_until_dice} ")
                for i in range(len(self.players_order) - 1):
                    f.write(f"{self.enemy_state[0][i]} {self.enemy_state[1][i]} {self.enemy_state[2][i]} ")
                f.write(f"\n")

        self.board_stats = [self.nb_players, self.biggest_region_size, self.hidden_regions, self.area_share, self.dice_share, self.dice_total, 
        self.region_count, self.dice_per_area, self.areas_on_border, self.areas_in_danger, self.avg_region_distance, self.players_ahead, self.turns_until_dice,
        self.enemy_state[0][0], self.enemy_state[1][0], self.enemy_state[2][0], self.enemy_state[0][1], self.enemy_state[1][1], self.enemy_state[2][1], self.enemy_state[0][2], self.enemy_state[1][2], self.enemy_state[2][2]]

        return self.board_stats


def convert_probability_to_classes(probability): # Converts float to one of: ["very low", "low", "medium", "high", "very high"]
    if probability < 0.15:
        return "very low"
    elif probability < 0.3:
        return "low"
    elif probability < 0.65:
        return "medium"
    else:
        return "high"

def convert_region_difference_to_classes(difference):
    if difference < 2:
        return "very low"
    elif difference < 4:
        return "low"
    elif difference < 7:
        return "medium"
    else:
        return "high"

def convert_neighbor_count_to_classes(count):
    if count == 1:
        return "one"
    elif count == 2:
        return "two"
    else:
        return "many"

def neighboring_field_count(board, area):
    return len(area.get_adjacent_areas())

def region_size_potential_gain(board, source, target, player_name):
    player_areas = board.get_player_areas(player_name) # Get all areas belonging to player
    player_areas = [area.get_name() for area in player_areas] # Convert area objects to names

    # Now to calculate true potential gain, we need to find the size of attacking region
    attacking_field_size = len(board.get_areas_region(source, player_areas))
    total_size = len(board.get_areas_region(source, player_areas + [target.get_name()]))

    return min(total_size - attacking_field_size, attacking_field_size + 1) # We take the potential gain from perspective of the bigger region

def region_size_potential_destroy(board, source, target, player_name): # How much the enemy region is decreased after the move
    enemy_name = target.get_owner_name()

    biggest_region_size = len(max(board.get_players_regions(enemy_name), key=len))

    # Temporarily "give" the area to player
    target.set_owner(player_name)

    # Calculate the difference in region size before and after attack
    size_difference = biggest_region_size - len(max(board.get_players_regions(enemy_name), key=len))

    # Return the area to the enemy player
    target.set_owner(enemy_name)

    return size_difference

def load_moves_from_game(snapshot_path):
    with shelve.open(snapshot_path + "moves.pickle", "c") as f:
        return f["moves"]

def load_parameters(snapshot_path):
    with shelve.open(snapshot_path + "config.pickle", "c") as f:
        return f["parameters"]

def save_parameters(snapshot_path, lr, epsilon, discount):
    with shelve.open(snapshot_path + "config.pickle", "c") as f:
        f["parameters"] = [lr, epsilon, discount]

def save_move_to_file(key, moves_path):
    with shelve.open(moves_path, "c", writeback=True) as f:
        if "moves" not in f:
            f["moves"] = []
        f["moves"].append(key)

def simulate_attack(board, turn):
    # Simulate whether attack succeeded
    source_area = board.get_area(turn.source_name)
    if random.uniform(0, 1) < probability_of_successful_attack(board, turn.source_name, turn.target_name):
        target_area = board.get_area(turn.target_name)
        target_area.set_owner(source_area.get_owner_name())
        target_area.set_dice(source_area.get_dice() - 1)

    source_area.set_dice(1)

    return board

def give_new_dice(board, players):
    for player in players:
        player_regions = board.get_players_regions(player)
        new_dice_to_give = len(max(player_regions, key=len))

        player_areas = board.get_player_areas(player)
        while new_dice_to_give > 0:
            area = random.choice(player_areas)

            if area.get_dice() != 8:
                area.set_dice(area.get_dice() + 1)
                new_dice_to_give -= 1
            
            if area.get_dice() == 8: # Remove it from the list if it's full
                player_areas.remove(area)
            
            if not player_areas: # If there's nowhere to give, break (we don't need to add dice to backup dice count)
                break

def give_reward_to_better_turns(q_table, reward, learning_rate, key, state, classes, start=True): # Noticed that turns with same risks but better payoffs get neglected during training because they are not played often
    if reward != 0:
        initial_value = key[0][state] # Potential field gain
        reward_multiplier = abs(reward * 0.01)
        if start: # Just so we give reward to first key only once
            start_index = classes.index(initial_value)
        else:
            start_index = classes.index(initial_value) + 1

        for i in classes[start_index:]: # Select only better classes
            key_list = [list(x) for x in key] # Convert tuple of tuples to list of lists so we can edit it
            key_list[0][state] = i
            key = tuple([tuple(x) for x in key_list]) # Revert
            
            if key in q_table:
                q_table[key] = q_table[key] + learning_rate * (reward + reward_multiplier) # Reward multiplier, working for both positive and negative rewards
                if key[1][0] == "attack": # Attack rewards increase and defend rewards decrease
                    reward_multiplier *= 1.01
                else:
                    reward_multiplier *= 0.99
            else: # Guarantees that all moves are created
                q_table[key] = 0

            if state < 3:
                q_table = give_reward_to_better_turns(q_table, reward + reward_multiplier, learning_rate, key, 3, ["very low", "low", "medium", "high"], start=False)
            #if state < 4:
            #    q_table = give_reward_to_better_turns(q_table, reward + reward_multiplier, learning_rate, key, 4, ["many", "two", "one"], start=False)
    return q_table

def calculate_risk_reward_bonus(key, reward): # Add reward based on riskiness of moves
    if key[1][0] == "attack":
        # Chance of winning
        if key[0][0] == "very low":
            reward -= 4
        elif key[0][0] == "low":
            reward -= 2
        elif key[0][0] == "high":
            reward += 2

        # Field hold chance
        if key[0][0] == "very low":
            reward -= 2
        elif key[0][0] == "low":
            reward -= 1
        elif key[0][0] == "high":
            reward += 1

    if key[1][0] == "defend":
        # Chance of winning
        if key[0][0] == "very low":
            reward += 4
        elif key[0][0] == "low":
            reward += 2
        elif key[0][0] == "high":
            reward -= 2

        # Field hold chance
        if key[0][0] == "very low":
            reward += 2
        elif key[0][0] == "low":
            reward += 1
        elif key[0][0] == "high":
            reward -= 1
    
    return reward

def hidden_region_count(board, player_name):
    areas = board.get_player_areas(player_name)
    areas_on_border = board.get_player_border(player_name)

    return len(areas) - len(areas_on_border)

def get_areas_in_danger(board, player_name):
    areas_in_danger = []
    for area in board.get_player_border(player_name):
        neighbours = area.get_adjacent_areas()

        for adj in neighbours:
            adjacent_area = board.get_area(adj)
            if adjacent_area.get_owner_name() != player_name:
                if adjacent_area.get_dice() > area.get_dice(): # Neighboring enemy area has atleast 1 more dice than owned area
                    areas_in_danger.append(area)
                    break
    
    return areas_in_danger

def get_most_opponent_regions(board, player_name, players_order):
    max_regions = 0
    for player in players_order:
        if player != player_name:
            areas = board.get_player_areas(player)

            if len(areas) > max_regions:
                max_regions = len(areas)
    
    return max_regions

def area_distance(board, source, target):
    visited_areas = [source.get_name()]
    areas_to_visit = source.get_adjacent_areas()
    new_areas_to_visit = []
    distance = 1

    while areas_to_visit: # While there are areas to visit
        for area in areas_to_visit:
            visited_areas.append(area)
            
            new_areas = board.get_area(area).get_adjacent_areas()

            for new_area in new_areas: 
                if new_area not in visited_areas and new_area not in areas_to_visit: # If area hasn't been already visited or isn't supposed to be, add it into "new_areas_to_visit" list
                    if new_area not in new_areas_to_visit: # We don'ť need to visit the same area twice
                        new_areas_to_visit.append(new_area)
        
        if target.get_name() in new_areas_to_visit: # If target area will be found next turn, return the distance
            return distance

        # Target area wasn't found in current list, so the distance is bigger (BFS) 
        distance += 1

        areas_to_visit = new_areas_to_visit
        new_areas_to_visit = []
    
    return distance

def average_region_distance(board, player_name):
    regions = board.get_players_regions(player_name)

    region_combinations = list(combinations(range(len(regions)), 2)) # All combinations of regions

    total_distance_between_regions = 0
    for source_idx, target_idx in region_combinations:
        min_distance = 99
        source_reg = regions[source_idx]
        target_reg = regions[target_idx]
        for source_area in source_reg:
            for target_area in target_reg:
                distance = area_distance(board, board.get_area(source_area), board.get_area(target_area))

                if min_distance > distance:
                    min_distance = distance

        total_distance_between_regions += min_distance

    if len(region_combinations) == 0: # We have only one region => there is no distance
        return 0
    else:
        return total_distance_between_regions / len(region_combinations)

def get_enemy_state(board, player_name, players_order):
    region_size = np.array([], dtype=int)
    area_count = np.array([], dtype=int)
    dice_count = np.array([], dtype=int)

    for player in players_order:
        if player != player_name:
            region_size = np.append(region_size, len(max(board.get_players_regions(player), key=len)))
            area_count = np.append(area_count, len(board.get_player_areas(player)))
            dice_count = np.append(dice_count, board.get_player_dice(player))
    
    for i in range(len(players_order) - 1, 8): # Fill up the rest of arrays with zeroes
        region_size = np.append(region_size, 0)
        area_count = np.append(area_count, 0)
        dice_count = np.append(dice_count, 0)
    
    # Approximate who has currently the best chance to win based on these 3 factors
    sort_array = region_size / dice_count * area_count
    sort_array = np.nan_to_num(sort_array) # Convert nan to zero

    # Get indexes required to make previous lists sorted based on this approximated array
    sort_index = np.flip(np.argsort(sort_array)) # Flip it so we get it in descending order

    # Sort all arrays and convert them to python lists
    region_size = region_size[sort_index].tolist()
    area_count = area_count[sort_index].tolist()
    dice_count = dice_count[sort_index].tolist()

    return [region_size, area_count, dice_count]

def turns_until_end_of_round(player_name, players_order):
    idx = players_order.index(player_name)
    
    return len(players_order) - 1 - idx

# How much will biggest region shrink if we lose the source area
def region_size_put_at_risk(board, source, target, player_name):
    prev_region_size = len(max(board.get_players_regions(player_name), key=len))

    # Temporarily take the area from player
    source.set_owner(target.get_owner_name())

    new_region_size = len(max(board.get_players_regions(player_name), key=len))

    # Return the area to original owner
    source.set_owner(player_name)

    return prev_region_size - new_region_size