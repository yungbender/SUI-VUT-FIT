import logging
from ..utils import possible_attacks

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand


class AI:
    """Agent using Strength Difference Checking (SDC) strategy

    This agent prefers moves with highest strength difference
    and doesn't make moves against areas with higher strength.
    """
    def __init__(self, player_name, board, players_order):
        """
        Parameters
        ----------
        game : Game

        Attributes
        ----------
            Areas that can make an attack
        """
        self.player_name = player_name
        self.logger = logging.getLogger('AI')

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn

        Creates a list with all possible moves along with associated strength
        difference. The list is then sorted in descending order with respect to
        the SD. A move with the highest SD is then made unless the highest
        SD is lower than zero - in this case, the agent ends its turn.
        """

        attacks = []
        for source, target in possible_attacks(board, self.player_name):
            area_dice = source.get_dice()
            strength_difference = area_dice - target.get_dice()
            attack = [source.get_name(), target.get_name(), strength_difference]
            attacks.append(attack)

        attacks = sorted(attacks, key=lambda attack: attack[2], reverse=True)

        if attacks and attacks[0][2] >= 0:
            return BattleCommand(attacks[0][0], attacks[0][1])

        return EndTurnCommand()
