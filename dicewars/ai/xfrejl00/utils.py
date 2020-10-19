from dicewars.client.game.board import Board
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import probability_of_successful_attack
import pickle
import random

def convert_probability_to_classes(probability): # Converts float to one of: ["very low", "low", "medium", "high", "very high"]
    if probability < 0.15:
        return "very low"
    elif probability < 0.3:
        return "low"
    elif probability < 0.65:
        return "medium"
    elif probability < 0.9:
        return "high"
    else:
        return "very high"


def region_size_potential_gain(board, source, target, player_name):
    player_areas = board.get_player_areas(player_name) # Get all areas belonging to player
    player_areas = [area.get_name() for area in player_areas] # Convert area objects to names

    # Now to calculate true potential gain, we need to find the size of attacking region
    attacking_field_size = len(board.get_areas_region(source, player_areas))
    total_size = len(board.get_areas_region(source, player_areas + [target.get_name()]))

    return min(total_size - attacking_field_size, attacking_field_size + 1) # We take the potential gain from perspective of the bigger region

def load_moves_from_game(snapshot_path):
    with open(snapshot_path + "moves.pickle", 'rb') as f:
        return pickle.load(f)

def load_parameters(snapshot_path):
    with open(snapshot_path + "config.pickle", "rb") as f:
        return pickle.load(f)

def save_parameters(snapshot_path, lr, epsilon, discount):
    with open(snapshot_path + "config.pickle", 'wb') as f: 
        pickle.dump([lr, epsilon, discount], f)

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

def give_reward_to_better_turns(q_table, reward, key): # Noticed that turns with same risks but better payoffs get neglected during training because they are not played often
    if key[1][0] == "attack" and reward != 0:
        initial_value = key[0][2] # Potential field gain
        reward_multiplier = abs(reward * 0.01)
        for i in range(initial_value+1, 16): # 16 should be potential maximal region gain
            key_list = [list(x) for x in key] # Convert tuple of tuples to list of lists so we can edit it
            key_list[0][2] = i
            key = tuple([tuple(x) for x in key_list]) # Revert
            
            if key in q_table:
                q_table[key] = q_table[key] + reward + reward_multiplier # Reward multiplier, working for both positive and negative rewards
                reward_multiplier *= 1.01 
    return q_table