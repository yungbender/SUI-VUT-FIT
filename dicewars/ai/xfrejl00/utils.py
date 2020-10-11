from dicewars.client.game.board import Board


def convert_probability_to_classes(probability): # Converts float to one of: ["very low", "low", "medium", "high", "very high"]
    if probability < 0.1:
        return "very low"
    elif probability < 0.3:
        return "low"
    elif probability < 0.7:
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