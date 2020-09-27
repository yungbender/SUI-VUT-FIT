import logging
from ..utils import probability_of_holding_area, probability_of_successful_attack
from ..utils import possible_attacks

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand


class AI:
    """Agent using improved Signle Turn Expectiminimax (STEi) strategy

    This agent makes such moves that have a probability of successful
    attack and hold over the area until next turn higher a 20% in two-player
    gams and higher than 40% in four-player games. In addition, it prefers
    attacks initiated from its largest region.
    """
    def __init__(self, player_name, board, players_order):
        """
        Parameters
        ----------
        game : Game

        Attributes
        ----------
        treshold : float
            Probability treshold for choosing an attack
        score_weight: float
            Preference of an attack from largest region over other attacks
        """
        self.player_name = player_name
        self.logger = logging.getLogger('AI')

        nb_players = board.nb_players_alive()
        self.logger.info('Setting up for {}-player game'.format(nb_players))
        if nb_players == 2:
            self.treshold = 0.2
            self.score_weight = 3
        else:
            self.treshold = 0.4
            self.score_weight = 2

        self.largest_region = []

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn

        Agent gets a list preferred moves and makes such move that has the
        highest estimated hold probability, prefering moves initiated from within
        the largest region. If there is no such move, the agent ends it's turn.
        """
        self.board = board
        self.logger.debug("Looking for possible turns.")
        self.get_largest_region()
        turns = self.possible_turns()

        if turns:
            turn = turns[0]
            self.logger.debug("Possible turn: {}".format(turn))
            hold_prob = turn[3]
            self.logger.debug("{0}->{1} attack and hold probabiliy {2}".format(turn[0], turn[1], hold_prob))

            return BattleCommand(turn[0], turn[1])

        self.logger.debug("No more plays.")
        return EndTurnCommand()

    def possible_turns(self):
        """Find possible turns with hold higher hold probability than treshold

        This method returns list of all moves with probability of holding the area
        higher than the treshold or areas with 8 dice. In addition, it includes
        the preference of these moves. The list is sorted in descending order with
        respect to preference * hold probability
        """
        turns = []
        for source, target in possible_attacks(self.board, self.player_name):
            atk_power = source.get_dice()
            atk_prob = probability_of_successful_attack(self.board, source.get_name(), target.get_name())
            hold_prob = atk_prob * probability_of_holding_area(self.board, target.get_name(), atk_power - 1, self.player_name)
            if hold_prob >= self.treshold or atk_power == 8:
                preference = hold_prob
                if source.get_name() in self.largest_region:
                    preference *= self.score_weight
                turns.append([source.get_name(), target.get_name(), preference, hold_prob])

        return sorted(turns, key=lambda turn: turn[2], reverse=True)

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

        self.largest_region = max_sized_regions[0]
        return max_region_size
