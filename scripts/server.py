#!/usr/bin/env python3

from argparse import ArgumentParser
import logging
import random

from itertools import cycle

from dicewars.server.board import Board
from dicewars.server.generator import BoardGenerator
from dicewars.server.game import Game


from utils import get_logging_level


def area_player_mapping(nb_players, nb_areas):
    assignment = {}
    unassigned_areas = list(range(1, nb_areas+1))
    player_cycle = cycle(range(1, nb_players+1))

    while unassigned_areas:
        player_no = next(player_cycle)
        area_no = random.choice(unassigned_areas)
        assignment[area_no] = player_no
        unassigned_areas.remove(area_no)

    return assignment


def continuous_area_player_mapping(nb_players, board):
    assignment = {}
    nb_areas = board.get_number_of_areas()
    unassigned_areas = set(range(1, nb_areas+1))
    player_cycle = cycle(range(1, nb_players+1))

    def unassigned_neighbours(area):
        return {area for area in board.get_area_by_name(area_no).get_adjacent_areas_names() if area in unassigned_areas}
        

    player_available = dict()
    for player_no in range(1, nb_players+1):
        area_no = random.choice(list(unassigned_areas))
        assignment[area_no] = player_no
        unassigned_areas.remove(area_no)
        player_available[player_no] = unassigned_neighbours(area_no)

    while unassigned_areas:
        player_no = next(player_cycle)
        player_available[player_no] &= unassigned_areas
        if player_available[player_no]:
            area_no = random.choice(list(player_available[player_no]))
        else:
            print(f"Having to start a new region for player {player_no}")
            area_no = random.choice(list(unassigned_areas))

        assignment[area_no] = player_no
        unassigned_areas.remove(area_no)
        if player_available[player_no]:
            player_available[player_no].remove(area_no)

        player_available[player_no] |= unassigned_neighbours(area_no)

    return assignment


def players_areas(ownership, the_player):
    return [area for area, player in ownership.items() if player == the_player]


def assign_dice_flat(board, nb_players, ownership):
    for area in board.areas.values():
        area.set_dice(3)


def assign_dice(board, nb_players, ownership):
    dice_total = 3 * board.get_number_of_areas() - random.randint(0, 5)
    players_processed = 0

    for player in range(1, nb_players+1):
        player_dice = int(round(dice_total / (nb_players - players_processed)))
        dice_total -= player_dice

        available_areas = [board.get_area_by_name(area_name) for area_name in players_areas(ownership, player)]

        # each area has to have at least one die
        for area in available_areas:
            area.set_dice(1)
            player_dice -= 1

        while player_dice and available_areas:
            area = random.choice(available_areas)
            if not area.add_die():  # adding a die to area failed means that area is full
                available_areas.remove(area)
            else:
                player_dice -= 1

        players_processed += 1


def main():
    """
    Server for Dice Wars
    """

    parser = ArgumentParser(prog='Dice_Wars-server')
    parser.add_argument('-n', '--number-of-players', help="Number of players", type=int, default=2)
    parser.add_argument('-p', '--port', help="Server port", type=int, default=5005)
    parser.add_argument('-a', '--address', help="Server address", default='127.0.0.1')
    parser.add_argument('-d', '--debug', help="Enable debug output", default='WARN')
    parser.add_argument('-b', '--board', help="Random seed to be used for board creating", type=int)
    parser.add_argument('-o', '--ownership', help="Random seed to be used for province assignment", type=int)
    parser.add_argument('-s', '--strength', help="Random seed to be used for dice assignment", type=int)
    parser.add_argument('-f', '--fixed', help="Random seed to be used for player order and dice rolls", type=int)
    parser.add_argument('--area-assignment', help="Method of assigning areas to players", choices=['orig', 'continuous'], default='orig')
    parser.add_argument('--dice-assignment', help="Method of assigning dice to areas", choices=['orig', 'flat'], default='orig')
    parser.add_argument('-r', '--order', nargs='+',
                        help="Random seed to be used for dice assignment")
    args = parser.parse_args()
    log_level = get_logging_level(args)

    logging.basicConfig(level=log_level)
    logger = logging.getLogger('SERVER')
    logger.debug("Command line arguments: {0}".format(args))

    random.seed(args.board)
    generator = BoardGenerator()
    board = Board(generator.generate_board(30))

    random.seed(args.ownership)
    if args.area_assignment == 'orig':
        area_ownership = area_player_mapping(args.number_of_players, board.get_number_of_areas())
    elif args.area_assignment == 'continous':
        area_ownership = continuous_area_player_mapping(args.number_of_players, board)

    random.seed(args.strength)

    if args.dice_assignment == 'orig':
        assign_dice(board, args.number_of_players, area_ownership)
    elif args.dice_assignment == 'flat':
        assign_dice_flat(board, args.number_of_players, area_ownership)

    random.seed(args.fixed)
    game = Game(board, area_ownership, args.number_of_players, args.address, args.port, args.order)
    game.run()


if __name__ == '__main__':
    main()
