import random
import logging
import time

from dicewars.ai.utils import possible_attacks
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand


class AI:
    """Naive player agent

    This agent performs all possible moves in random order
    """

    def __init__(self, player_name, board, players_order):
        """
        Parameters
        ----------
        game : Game
        """
        self.logger = logging.getLogger('AI')
        self.player_name = player_name

        sleep_len = random.uniform(0.1, 0.3)
        self.logger.debug("Taking a constructive sleep of len {:.3f}".format(sleep_len))
        time.sleep(sleep_len)

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn

        Get a random area. If it has a possible move, the agent will do it.
        If there are no more moves, the agent ends its turn.
        """
        sleep_len = random.uniform(0.1, 0.3)
        self.logger.debug("Having {:.3f}s, taking sleep of len {:.3f}".format(time_left, sleep_len))
        time.sleep(sleep_len)

        if nb_moves_this_turn == 2:
            self.logger.debug("I'm too well behaved. Let others play now.")
            return EndTurnCommand()

        attacks = list(possible_attacks(board, self.player_name))
        if attacks:
            source, target = random.choice(attacks)
            return BattleCommand(source.get_name(), target.get_name())
        else:
            self.logger.debug("No more possible turns.")
            return EndTurnCommand()
