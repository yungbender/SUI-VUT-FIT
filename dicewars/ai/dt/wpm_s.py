import numpy
import logging

from ..utils import probability_of_successful_attack, sigmoid
from ..utils import possible_attacks

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand


class AI:
    """Agent using Win Probability Maximization (WPM) using player scores

    This agent estimates win probability given the current state of the game.
    As a feature to describe the state, a vector of players' scores is used.
    The agent choses such moves, that will have the highest improvement in
    the estimated probability.
    """
    def __init__(self, player_name, board, players_order):
        """
        Parameters
        ----------
        game : Game

        Attributes
        ----------
        players_order : list of int
            Names of players in the order they are playing, with the agent being first
        weights : dict of numpy.array
            Weights for estimating win probability
        largest_region: list of int
            Names of areas in the largest region
        """
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.players = board.nb_players_alive()

        self.largest_region = []

        self.players_order = players_order
        while self.player_name != self.players_order[0]:
            self.players_order.append(self.players_order.pop(0))

        self.weights = {
            2: numpy.array([0.51862355, -0.417179]),
            3: numpy.array([0.24112347, -0.20702862, -0.20097175]),
            4: numpy.array([0.26457488, -0.20733951, -0.19326027, -0.20171941]),
            5: numpy.array([0.26777938, -0.1878346, -0.18560973, -0.20005864, -0.18976791]),
            6: numpy.array([0.2700982, -0.18000744, -0.18290534, -0.1815374, -0.20105069, -0.1808327]),
            7: numpy.array([0.27109102, -0.18051686, -0.18232428, -0.17905882, -0.17959111, -0.17958394, -0.17634735]),
            8: numpy.array([0.277179, -0.16852433, -0.18678373, -0.17492631, -0.17996621, -0.1790844, -0.16977776, -0.18876063]),
        }[self.players]

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn

        This agent estimates probability to win the game from the feature vector associated
        with the outcome of the move and chooses such that has highest improvement in the
        probability.
        """
        self.board = board
        self.logger.debug("Looking for possible turns.")
        turns = self.possible_turns()

        if turns:
            turn = turns[0]
            self.logger.debug("Possible turn: {}".format(turn))
            atk_area = self.board.get_area(turn[0])
            atk_power = atk_area.get_dice()

            if turn[2] >= -0.05 or atk_power == 8:
                return BattleCommand(turn[0], turn[1])

        self.logger.debug("No more plays.")
        return EndTurnCommand()

    def possible_turns(self):
        """Get list of possible turns with the associated improvement
        in estimated win probability
        """
        turns = []

        features = []
        for p in self.players_order:
            features.append(self.get_score_by_player(p))
        win_prob = numpy.log(sigmoid(numpy.dot(numpy.array(features), self.weights)))

        self.get_largest_region()

        for source, target in possible_attacks(self.board, self.player_name):
            area_name = source.get_name()
            atk_power = source.get_dice()

            opponent_name = target.get_owner_name()

            increase_score = False
            if area_name in self.largest_region:
                increase_score = True
            else:
                for n in target.get_adjacent_areas():
                    if n in self.largest_region:
                        increase_score = True
                        break

            if increase_score or atk_power == 8:
                atk_prob = numpy.log(probability_of_successful_attack(self.board, area_name, target.get_name()))
                new_features = []
                for p in self.players_order:
                    idx = self.players_order.index(p)
                    if p == self.player_name:
                        new_features.append(features[idx] + 1 if increase_score else features[idx])
                    elif p == opponent_name:
                        new_features.append(self.get_score_by_player(p, skip_area=target.get_name()))
                    else:
                        new_features.append(features[idx])
                new_win_prob = numpy.log(sigmoid(numpy.dot(numpy.array(new_features), self.weights)))
                total_prob = new_win_prob + atk_prob
                improvement = total_prob - win_prob
                if improvement >= -1:
                    turns.append([area_name, target.get_name(), improvement])

        return sorted(turns, key=lambda turn: turn[2], reverse=True)

    def get_score_by_player(self, player_name, skip_area=None):
        """Get score of a player

        Parameters
        ----------
        player_name : int
        skip_area : int
            Name of an area to be excluded from the calculation

        Returns
        -------
        int
            score of the player
        """
        players_regions = self.board.get_players_regions(self.player_name, skip_area=skip_area)
        max_region_size = max(len(region) for region in players_regions)

        return max_region_size

    def get_largest_region(self):
        """Get size of the largest region, including the areas within

        Attributes
        ----------
        largest_region : list of int
            Names of areas in the largest region

        Returns
        -------
        int
            Number of areas in the largest region
        """
        self.largest_region = []

        players_regions = self.board.get_players_regions(self.player_name)
        max_region_size = max(len(region) for region in players_regions)
        max_sized_regions = [region for region in players_regions if len(region) == max_region_size]

        for region in max_sized_regions:
            for area in region:
                self.largest_region.append(area)
        return max_region_size
