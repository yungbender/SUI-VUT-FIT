import os
import sys
from subprocess import Popen
import tempfile
import numpy as np
import random

from dicewars.server.summary import GameSummary


class BoardDefinition:
    def __init__(self, board, ownership, strength):
        assert(board is None or isinstance(board, int))
        assert(ownership is None or isinstance(ownership, int))
        assert(strength is None or isinstance(strength, int))
        self.board = board
        self.ownership = ownership
        self.strength = strength

    def to_args(self):
        args = []
        if self.board is not None:
            args.extend(['-b', str(self.board)])
        if self.ownership is not None:
            args.extend(['-o', str(self.ownership)])
        if self.strength is not None:
            args.extend(['-s', str(self.strength)])
        return args

    def __str__(self):
        return "board: {}, ownership: {}, strength: {}".format(self.board, self.ownership, self.strength)


def get_logging_level(args):
    """
    Parse command-line arguments.
    """
    if args.debug.lower() == 'debug':
        logging = 10
    elif args.debug.lower() == 'info':
        logging = 20
    elif args.debug.lower() == 'error':
        logging = 40
    else:
        logging = 30

    return logging


def get_nickname(ai_spec):
    if ai_spec is not None:
        nick = '{} (AI)'.format(ai_spec)
    else:
        nick = 'Human'

    return nick


def log_file_producer(logdir, process):
    if logdir is None:
        return open(os.devnull, 'w')
    else:
        return open('{}/{}'.format(logdir, process), 'w')


def run_ai_only_game(
        port, address, process_list, ais,
        board_definition=None, fixed=None, client_seed=None,
        logdir=None, debug=False):
    logs = []
    process_list.clear()

    ai_nicks = [get_nickname(ai) for ai in ais]

    server_cmd = [
        "./scripts/server.py",
        "-n", str(len(ais)),
        "-p", str(port),
        "-a", str(address),
        "--dice-assignment", 'flat',
    ]
    server_cmd.append('-r')
    server_cmd.extend(ai_nicks)
    if board_definition is not None:
        server_cmd.extend(board_definition.to_args())
    if fixed is not None:
        server_cmd.extend(['-f', str(fixed)])
    if debug:
        server_cmd.extend(['--debug', 'DEBUG'])

    server_output = tempfile.TemporaryFile('w+')
    logs.append(log_file_producer(logdir, 'server.txt'))
    process_list.append(Popen(server_cmd, stdout=server_output, stderr=logs[-1]))

    for ai_version in ais:
        client_cmd = [
            "./scripts/client.py",
            "-p", str(port),
            "-a", str(address),
            "--ai", str(ai_version),
        ]
        if client_seed is not None:
            client_cmd.extend(['-s', str(client_seed)])
        if debug:
            client_cmd.extend(['--debug', 'DEBUG'])

        logs.append(log_file_producer(logdir, 'client-{}.log'.format(ai_version)))
        process_list.append(Popen(client_cmd, stderr=logs[-1]))

    for p in process_list:
        p.wait()

    for log in logs:
        log.close()

    server_output.seek(0)
    game_summary = GameSummary.from_repr(server_output.read())
    return game_summary


class ListStats:
    def __init__(self, the_list):
        self.min = min(the_list)
        self.avg = sum(the_list)/len(the_list)
        self.max = max(the_list)

    def __str__(self):
        return 'min/avg/max {}/{:.2f}/{}'.format(self.min, self.avg, self.max)


class SingleLineReporter:
    def __init__(self, mute):
        self.last_line_len = 0
        self.mute = mute

    def clean(self):
        if self.mute:
            return

        sys.stdout.write('\r' + ' '*self.last_line_len + ' '*len('^C'))
        sys.stdout.write('\r')

    def report(self, line):
        if self.mute:
            return

        self.clean()
        self.last_line_len = len(line)
        sys.stdout.write(line)


class PlayerPerformance:
    def __init__(self, name, games, players):
        nickname = get_nickname(name)
        self.nb_games = len(games)
        self.nb_wins = sum(game.winner == nickname for game in games)
        self.players = players
        if self.nb_games > 0:
            self.winrate = self.nb_wins/self.nb_games
        else:
            self.winrate = float('nan')
        self.name = name

        self.per_competitor_winrate = {}
        for competitor in self.players:
            his_games = [game for game in games if get_nickname(competitor) in game.participants()]
            if his_games:
                self.per_competitor_winrate[competitor] = (sum(game.winner == nickname for game in his_games)/len(his_games), len(his_games))
            else:
                self.per_competitor_winrate[competitor] = (float('nan'), len(his_games))

    def __str__(self):
        per_competitor_str = ' '.join('{:.1f}/{}'.format(100.0*winrate[0], winrate[1]) for ai, winrate in self.per_competitor_winrate.items())
        return '{} {:.2f} % winrate [ {} / {} ] {}'.format(self.name, 100.0*self.winrate, self.nb_wins, self.nb_games, per_competitor_str)

    def competitors_header(self):
        return '{} {} % winrate [ {} / {} ] {}'.format('.', '.', '.', '.', ' '.join(str(ai) for ai in self.players))


class TournamentCombatantsProvider:
    def __init__(self, players):
        self.game_numbers = np.zeros((len(players), len(players)), dtype=np.int)
        self.players = players

    def get_combatants(self, nb_combatants):
        per_player_count = {ai: nb_games for ai, nb_games in zip(self.players, np.sum(self.game_numbers, axis=1))}

        least_playing = sorted(per_player_count, key=lambda p: per_player_count[p])[0]
        pivot_ind = self.players.index(least_playing)

        if self.game_numbers[pivot_ind][pivot_ind] == 0:
            rare_opponent_ind = (pivot_ind + 1) % len(self.players)
        else:
            rare_opponent_ind = np.argmin(self.game_numbers[pivot_ind])
        assert(rare_opponent_ind != pivot_ind)

        possible_competitors = [self.players.index(ai) for ai in self.players if self.players.index(ai) not in [pivot_ind, rare_opponent_ind]]
        random.shuffle(possible_competitors)
        competitors = possible_competitors[:nb_combatants-2]

        players = [pivot_ind, rare_opponent_ind] + competitors

        for a_ind in players:
            for b_ind in players:
                self.game_numbers[a_ind][b_ind] += 1

        return [self.players[p] for p in players]


class EvaluationCombatantsProvider:
    def __init__(self, players, ai_under_test):
        self.game_numbers = np.zeros((len(players), len(players)), dtype=np.int)
        self.players = players
        self.put = ai_under_test
        assert(self.put in self.players)

    def get_combatants(self, nb_combatants):
        pivot_ind = self.players.index(self.put)

        if self.game_numbers[pivot_ind][pivot_ind] == 0:
            rare_opponent_ind = (pivot_ind + 1) % len(self.players)
        else:
            rare_opponent_ind = np.argmin(self.game_numbers[pivot_ind])
        assert(rare_opponent_ind != pivot_ind)

        possible_competitors = [self.players.index(ai) for ai in self.players if self.players.index(ai) not in [pivot_ind, rare_opponent_ind]]
        random.shuffle(possible_competitors)
        competitors = possible_competitors[:nb_combatants-2]

        players = [pivot_ind, rare_opponent_ind] + competitors

        for a_ind in players:
            for b_ind in players:
                self.game_numbers[a_ind][b_ind] += 1

        return [self.players[p] for p in players]


def column_t(items):
    for line in items:
        assert(len(line) == len(items[0]))

    col_widths = []
    for col_id in range(len(items[0])):
        col_widths.append(max(len(line[col_id]) for line in items))

    formatted_lines = []
    for line in items:
        fmts = ['{{: <{}}}'.format(width) for width in col_widths]
        line_fmt = '{}\n'.format(' '.join(fmts))
        formatted_lines.append(line_fmt.format(*line))

    return ''.join(formatted_lines)
