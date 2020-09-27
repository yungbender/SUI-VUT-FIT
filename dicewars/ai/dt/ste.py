import logging
from ..utils import possible_attacks
from ..utils import probability_of_holding_area, probability_of_successful_attack

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand


class AI:
    """Agent using Single Turn Expectiminimax (STE) strategy

    This agent makes such moves that have a probability of successful
    attack and hold over the area until next turn higher than 20 %.
    """
    def __init__(self, player_name, board, players_order):
        """
        Parameters
        ----------
        game : Game
        """
        self.player_name = player_name
        self.logger = logging.getLogger('AI')

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn

        Agent gets a list preferred moves and makes such move that has the
        highest estimated hold probability. If there is no such move, the agent
        ends it's turn.
        """
        self.logger.debug("Looking for possible turns.")
        self.board = board
        turns = self.possible_turns()

        if turns:
            turn = turns[0]
            area_name = turn[0]
            self.logger.debug("Possible turn: {}".format(turn))
            hold_prob = turn[2]
            self.logger.debug("{0}->{1} attack and hold probabiliy {2}".format(area_name, turn[1], hold_prob))

            return BattleCommand(area_name, turn[1])

        self.logger.debug("No more plays.")
        return EndTurnCommand()

    def possible_turns(self):
        """Get a list of preferred moves

        This list is sorted with respect to hold probability in descending order.
        It includes all moves that either have hold probability higher or equal to 20 %
        or have strength of eight dice.
        """
        turns = []

        for source, target in possible_attacks(self.board, self.player_name):
            area_name = source.get_name()
            atk_power = source.get_dice()
            atk_prob = probability_of_successful_attack(self.board, area_name, target.get_name())
            hold_prob = atk_prob * probability_of_holding_area(self.board, target.get_name(), atk_power - 1, self.player_name)
            if hold_prob >= 0.2 or atk_power == 8:
                turns.append([area_name, target.get_name(), hold_prob])

        return sorted(turns, key=lambda turn: turn[2], reverse=True)
