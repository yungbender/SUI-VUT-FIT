from dicewars.client.game.board import Board
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import probability_of_successful_attack
import random
import shelve

def convert_probability_to_classes(probability): # Converts float to one of: ["very low", "low", "medium", "high", "very high"]
    if probability < 0.15:
        return "very low"
    elif probability < 0.3:
        return "low"
    elif probability < 0.65:
        return "medium"
    else: #if probability < 0.9:
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

def give_reward_to_better_turns(q_table, reward, learning_rate, key, state, classes): # Noticed that turns with same risks but better payoffs get neglected during training because they are not played often
    if reward != 0:
        initial_value = key[0][state] # Potential field gain
        reward_multiplier = abs(reward * 0.01)
        start_index = classes.index(initial_value)
        for i in classes[start_index+1:]: # Select only better classes
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
    return q_table

def calculate_risk_reward_multiplier(key, reward): # Add reward based on riskiness of moves
    if key[1][0] == "attack":
        # Chance of winning
        if key[0][0] == "very low":
            reward -= 1
        elif key[0][0] == "low":
            reward -= 0.5
        elif key[0][0] == "high":
            reward += 1

        # Field hold chance
        if key[0][0] == "very low":
            reward -= 0.5
        elif key[0][0] == "low":
            reward -= 0.25
        elif key[0][0] == "high":
            reward += 0.5

    if key[1][0] == "defend":
        # Chance of winning
        if key[0][0] == "very low":
            reward += 1
        elif key[0][0] == "low":
            reward += 0.5
        elif key[0][0] == "high":
            reward -= 1

        # Field hold chance
        if key[0][0] == "very low":
            reward += 0.5
        elif key[0][0] == "low":
            reward += 0.25
        elif key[0][0] == "high":
            reward -= 0.5
    
    return reward

def hidden_region_count(board, player_name):
    areas = board.get_player_areas(player_name)
    areas_on_border = board.get_player_border(player_name)

    return len(areas) - len(areas_on_border) 