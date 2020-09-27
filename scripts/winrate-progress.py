#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle


class PlayerRecord:
    def __init__(self):
        self.nb_games = 0
        self.nb_wins = 0
        self.entries = []
        self.game_stamps = []

    def score_game(self, game_no, win):
        if len(self.game_stamps) > 0:
            assert(game_no > self.game_stamps[-1])
        self.game_stamps.append(game_no)

        self.nb_games += 1
        if win:
            self.nb_wins += 1

        self.entries.append((self.nb_games, self.nb_wins))

    @property
    def winrates(self):
        return [100.0*wins/games for games, wins in self.entries]

    @property
    def final_winrate(self):
        return 100.0 * self.nb_wins / self.nb_games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xmin', type=int, default=0, help='how many game shall be skipped in the graph')
    parser.add_argument('games', help='where the games are stored')
    args = parser.parse_args()

    with open(args.games, 'rb') as f:
        games = pickle.load(f)

    players = {}
    nb_games_processed = 0
    for game in games:
        nb_games_processed += 1
        if game.winner not in players:
            players[game.winner] = PlayerRecord()
        players[game.winner].score_game(nb_games_processed, True)

        eliminated = [e[0] for e in game.eliminations]
        for loser in eliminated:
            if loser not in players:
                players[loser] = PlayerRecord()
            players[loser].score_game(nb_games_processed, False)

    plt.figure()
    for name, record in sorted(players.items(), key=lambda n_r: n_r[1].final_winrate, reverse=True):
        mask = np.asarray(record.game_stamps) > args.xmin
        plt.plot(
            np.asarray(record.game_stamps)[mask],
            np.asarray(record.winrates)[mask],
            label='{} ({:.1f} %)'.format(name, record.final_winrate),
            drawstyle='steps-pre'
        )

    plt.ylim(bottom=0)
    plt.xlim(left=args.xmin)

    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.show()


if __name__ == '__main__':
    main()
