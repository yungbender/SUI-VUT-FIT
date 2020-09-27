#!/usr/bin/env python3
import sys
from signal import signal, SIGCHLD
from argparse import ArgumentParser

from dicewars.server.summary import get_win_rates
from utils import run_ai_only_game, ListStats, BoardDefinition


parser = ArgumentParser(prog='Dice_Wars')
parser.add_argument('-n', '--nb-games', help="Number of games.", type=int, default=1)
parser.add_argument('-p', '--port', help="Server port", type=int, default=5005)
parser.add_argument('-a', '--address', help="Server address", default='127.0.0.1')
parser.add_argument('-b', '--board', help="Seed for generating board", type=int)
parser.add_argument('-s', '--strength', help="Seed for dice assignment", type=int)
parser.add_argument('-o', '--ownership', help="Seed for province assignment", type=int)
parser.add_argument('-f', '--fixed', help="Random seed to be used for player order and dice rolls", type=int)
parser.add_argument('-c', '--client-seed', help="Seed for clients", type=int)
parser.add_argument('-l', '--logdir', help="Folder to store last running logs in.")
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('--ai', help="Specify AI versions as a sequence of ints.", nargs='+')
parser.add_argument('-r', '--report', help="State the game number on the stdout", action='store_true')

procs = []


def signal_handler(signum, frame):
    """Handler for SIGCHLD signal that terminates server and clients
    """
    for p in procs:
        try:
            p.kill()
        except ProcessLookupError:
            pass


def main():
    """
    Run the Dice Wars game among AI's.

    Example:
        ./dicewars.py --nb-games 16 --ai 4 4 2 1
        # runs 16 games four-player games with AIs 4 (two players), 2, and 1
    """
    args = parser.parse_args()

    signal(SIGCHLD, signal_handler)

    if len(args.ai) < 2 or len(args.ai) > 8:
        print("Unsupported number of AIs")
        exit(1)

    summaries = []
    for i in range(args.nb_games):
        if args.report:
            sys.stdout.write('\r{}'.format(i))
        try:
            board_seed = None if args.board is None else args.board + i
            board_definition = BoardDefinition(board_seed, args.ownership, args.strength)
            game_summary = run_ai_only_game(
                args.port, args.address, procs, args.ai,
                board_definition,
                fixed=args.fixed,
                client_seed=args.client_seed,
                logdir=args.logdir,
                debug=args.debug,
            )
            summaries.append(game_summary)
        except KeyboardInterrupt:
            for p in procs:
                p.kill()
            break
        except AttributeError:
            for p in procs:
                p.kill()
    if args.report:
        sys.stdout.write('\r')

    win_numbers = get_win_rates(summaries, len(args.ai))
    sys.stdout.write("Win counts {}\n".format(win_numbers))

    nb_battles_stats = ListStats([s.nb_battles for s in summaries])
    sys.stdout.write("Nb battles {}\n".format(nb_battles_stats))


if __name__ == '__main__':
    main()
