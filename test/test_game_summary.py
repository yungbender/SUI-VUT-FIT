import unittest

from dicewars.server.game.summary import GameSummary


class GameSummaryTests(unittest.TestCase):
    def test_repr_loading(self):
        summary = GameSummary()
        summary.set_winner('joe')
        reconstructed = GameSummary.from_repr(repr(summary))
        self.assertEqual(repr(reconstructed), repr(summary))

    def test_repr_loading_with_battles(self):
        summary = GameSummary()
        summary.set_winner('joe')
        summary.add_battle()
        summary.add_battle()
        reconstructed = GameSummary.from_repr(repr(summary))

        self.assertEqual(repr(reconstructed), repr(summary))
        self.assertEqual(reconstructed.nb_battles, 2)

    def test_repr_loading_with_elimination(self):
        summary = GameSummary()
        summary.set_winner('joe')
        summary.add_battle()
        summary.add_elimination('looser', 1)
        summary.add_battle()
        summary.add_elimination('mediocore', 2)
        reconstructed = GameSummary.from_repr(repr(summary))

        self.assertEqual(repr(reconstructed), repr(summary))
        self.assertEqual(reconstructed.nb_battles, 2)
