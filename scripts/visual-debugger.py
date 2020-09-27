#!/usr/bin/env python3
import argparse
from PyQt5.QtWidgets import QApplication
import sys

from dicewars.client.game.debugger_game import StaticGame
from dicewars.client import debugger_ui


from dicewars.ai.xlogin42.utils import attacker_advantage


class DetailedAreaReporter:
    def __init__(self, board):
        self.board = board

    def __call__(self, area):
        neighbours = [self.board.get_area(a) for a in area.get_adjacent_areas()]
        enemy_neighbours = [a for a in neighbours if a.get_owner_name() != area.get_owner_name()]
        return '{}: {} -- {}\n'.format(
            area.get_name(),
            [n.get_name() for n in neighbours],
            [(n.get_name(), attacker_advantage(area, n)) for n in enemy_neighbours],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('savegame')
    args = parser.parse_args()

    with open(args.savegame, 'rb') as f:
        game = StaticGame(f)

    area_describer = DetailedAreaReporter(game.board)
    debugger_ui.on_area_activation = area_describer

    app = QApplication(sys.argv)
    ui = debugger_ui.DebuggerUI(game)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
