#!/usr/bin/env python3
from signal import signal, SIGCHLD
from argparse import ArgumentParser

import math
import itertools
from utils import run_ai_only_game, get_nickname, BoardDefinition, SingleLineReporter, PlayerPerformance
from utils import TournamentCombatantsProvider, EvaluationCombatantsProvider
from utils import column_t
import random
import sys
import pickle


parser = ArgumentParser(prog='Dice_Wars')
parser.add_argument('-p', '--port', help="Server port", type=int, default=5005)
parser.add_argument('-a', '--address', help="Server address", default='127.0.0.1')
parser.add_argument('-b', '--board', help="Seed for generating board", type=int)
parser.add_argument('-n', '--nb-boards', help="How many boards should be played", type=int, required=True)
parser.add_argument('-g', '--game-size', help="How many players should play a game", type=int, required=True)
parser.add_argument('-s', '--seed', help="Seed sampling players for a game", type=int)
parser.add_argument('-l', '--logdir', help="Folder to store last running logs in.")
parser.add_argument('--ai-under-test', help="Only play this AI against others")
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-r', '--report', help="State the game number on the stdout", action='store_true')
parser.add_argument('--save', help="Where to put pickled GameSummaries")
parser.add_argument('--load', help="Which GameSummaries to start from")

procs = []


def signal_handler(signum, frame):
    """Handler for SIGCHLD signal that terminates server and clients
    """
    for p in procs:
        try:
            p.kill()
        except ProcessLookupError:
            pass


PLAYING_AIs = [
    'dt.rand',
    'dt.sdc',
    'dt.ste',
    # 'dt.stei',
    # 'dt.wpm_d',
    # 'dt.wpm_s',
    'dt.wpm_c',
    # 'xlogin42',
    'xlogin00',
]
UNIVERSAL_SEED = 42

players_info = {ai: {'games': []} for ai in PLAYING_AIs}


def board_definitions(initial_board_seed):
    board_seed = initial_board_seed
    while True:
        yield BoardDefinition(board_seed, UNIVERSAL_SEED, UNIVERSAL_SEED)
        board_seed += 1


def full_permunations_generator(players):
    nb_perms = math.factorial(len(players))
    perms_generator = itertools.permutations(players)

    return nb_perms, perms_generator


def rotational_permunations_generator(players):
    def all_rotations(a_list):
        for _ in range(len(a_list)):
            head = a_list[0]
            a_list = a_list[1:] + [head]
            yield a_list

    random.shuffle(players)
    return len(players), all_rotations(players)


def main():
    args = parser.parse_args()
    if args.ai_under_test is not None:
        combatants_provider = EvaluationCombatantsProvider(PLAYING_AIs, args.ai_under_test)
    else:
        combatants_provider = TournamentCombatantsProvider(PLAYING_AIs)
    random.seed(args.seed)

    signal(SIGCHLD, signal_handler)

    if args.load:
        with open(args.load, 'rb') as f:
            all_games = pickle.load(f)
    else:
        all_games = []

    boards_played = 0
    reporter = SingleLineReporter(not args.report)
    try:
        for board_definition in board_definitions(args.board):
            if boards_played == args.nb_boards:
                break
            boards_played += 1

            combatants = combatants_provider.get_combatants(args.game_size)
            nb_permutations, permutations_generator = rotational_permunations_generator(combatants)
            for i, permuted_combatants in enumerate(permutations_generator):
                reporter.report('\r{} {}/{} {}'.format(boards_played, i+1, nb_permutations, ' vs. '.join(permuted_combatants)))
                game_summary = run_ai_only_game(
                    args.port, args.address, procs, permuted_combatants,
                    board_definition,
                    fixed=UNIVERSAL_SEED,
                    client_seed=UNIVERSAL_SEED,
                    logdir=args.logdir,
                    debug=args.debug,
                )
                all_games.append(game_summary)
    except (Exception, KeyboardInterrupt) as e:
        sys.stderr.write("Breaking the tournament because of {}\n".format(repr(e)))
        for p in procs:
            p.kill()

    reporter.clean()

    if args.save:
        with open(args.save, 'wb') as f:
            pickle.dump(all_games, f)

    for game in all_games:
        participants = game.participants()
        for player in players_info:
            if get_nickname(player) in participants:
                players_info[player]['games'].append(game)

    performances = [PlayerPerformance(player, info['games'], PLAYING_AIs) for player, info in players_info.items()]
    performances.sort(key=lambda perf: perf.winrate, reverse=True)

    perf_strings = [performances[0].competitors_header()] + [str(perf) for perf in performances]
    fields = [perf.split() for perf in perf_strings]

    print(column_t(fields))


if __name__ == '__main__':
    main()
