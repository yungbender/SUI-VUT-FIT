import logging
import itertools

from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton

from .ui import Battle, MainWindow, Score, StatusArea

import sys


area_descriptors = [
    ('Dice', lambda area: str(area.get_dice())),
    ('Name', lambda area: str(area.get_name())),
    ('Owner', lambda area: str(area.get_owner_name())),
]


def on_area_activation(area_name):
    return ''


def descriptors_provider():
    for i in itertools.count(0):
        yield area_descriptors[i % len(area_descriptors)]


class DebuggerUI(QWidget):
    """Dice Wars' graphical user interface
    """
    def __init__(self, game):
        """
        Parameters
        ----------
        game : Game
        """
        super().__init__()
        self.logger = logging.getLogger('GUI')
        self.game = game
        self.window_name = 'Debugger - Player ' + str(self.game.player_name)
        self.init_ui()

        self.area_text_fn_it = descriptors_provider()
        self.handle_change_labels_button()

        self.game.battle = False
        self.game.draw_battle = False

    def init_ui(self):
        self.resize(1024, 576)
        self.setMinimumSize(1024, 576)
        self.setWindowTitle(self.window_name)

        self.init_layout()
        self.show()

    def init_layout(self):
        grid = QGridLayout()

        self.main_area = MainWindow(self.game, lambda area: str(area.get_name()))
        self.main_area.mousePressEvent = mousePressEvent_monkeypatch.__get__(self.main_area)

        self.battle_area = Battle(self.game)
        self.score_area = Score(self.game)
        self.status_area = StatusArea(self.game)

        self.change_labels = QPushButton('Relabel')
        self.change_labels.clicked.connect(self.handle_change_labels_button)
        self.change_labels.setEnabled(True)

        grid.addWidget(self.main_area, 0, 0, 10, 8)
        grid.addWidget(self.battle_area, 0, 8, 4, 3)
        grid.addWidget(self.score_area, 4, 8, 4, 3)
        grid.addWidget(self.change_labels, 8, 9, 1, 1)
        grid.addWidget(self.status_area, 9, 8, 1, 3)

        self.setLayout(grid)

    def handle_change_labels_button(self):
        name, fn = next(self.area_text_fn_it)
        self.main_area.set_area_text_fn(fn)
        self.main_area.update()
        self.change_labels.setText(name)


def mousePressEvent_monkeypatch(self, event):
    hexagon = self.get_hex(event.pos())
    try:
        area = self.board.get_area(self.areas_mapping[hexagon])

        if self.activated_area_name:
            if area.get_name() == self.activated_area_name:
                self.activated_area_name = None
                self.update()
        else:
            self.activated_area_name = area.get_name()
            self.activated_area = area
            self.update()
            sys.stdout.write(on_area_activation(self.activated_area))
    except KeyError:
        pass
