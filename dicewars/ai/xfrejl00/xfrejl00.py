import logging
from datetime import datetime
import tensorflow as tf

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import save_state


class AlphaDice:
	def __init__(self, player_name, board, players_order):
		self.player_name = player_name
		self.players_order = players_order
		self.logger = logging.getLogger('AI')
		self.logger.info("Current time: " + datetime.now().strftime('%Y.%m.%d %H:%M:%S'))
		self.logger.info("Tensorflow version: " + tf.__version__)
		
	def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
		with open("xfrejl00.save", "wb") as f:
			save_state(f, board, self.player_name, self.players_order)

		self.logger.debug("Player areas: ")
		for player in self.players_order:
			self.logger.debug(str(board.get_player_areas(player)))

		self.logger.debug("Player borders: ")
		for player in self.players_order:
			self.logger.debug(str(board.get_player_border(player)))

		self.logger.debug("Player regions: ")
		for player in self.players_order:
			self.logger.debug(str(board.get_players_regions(player)))

		return EndTurnCommand()