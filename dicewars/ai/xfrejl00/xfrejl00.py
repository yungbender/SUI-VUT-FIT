import logging
from datetime import datetime
from configparser import ConfigParser

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import save_state


class AlphaDice:
    def __init__(self, player_name, board, players_order, update_qtable=False):
        config = ConfigParser()
        config.read('dicewars/ai/xfrejl00/config.ini')

        self.train = config.getboolean('BASE', 'Train')
        self.snapshot_path = config['BASE']['SnapshotPath']
        self.player_name = player_name
        self.players_order = players_order
        self.update_qtable = update_qtable
        self.logger = logging.getLogger('AI')
        self.logger.info("Current time: " + datetime.now().strftime('%Y.%m.%d %H:%M:%S'))
        # TODO: Load the Q-table (default: from xfrejl00 folder, also that's where we will store the final q-table)

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

        """
        TODO: State definition
          - Friendly dice count 
          - Enemy dice count (or probability of winning to replace both (very low, low, medium, high, very high)?)
          - Region size potential gain
          - Probability of keeping the area until the next turn (very low, low, medium, high, very high)?

        TODO: Parse the board data and provide "state" for Q-table
        TODO: Decide whether random action will be made (if self.update_qtable == True)
        for move in available_moves:
            for action in ["attack", "defend"]:
                value = evaluate_move([state,action])
                if value > value_max:
                    value_max = value
                    action_max = action
                    move_max = move
        
        TODO: Save the chosen move so rewards received at the end of the game can be added too (if self.update_qtable == True)
        TODO: Update Q-table based on Bellman equation with immediate rewards (if self.update_qtable == True)
        TODO: Perform action that was obtained from Q-table
        """

        return EndTurnCommand()

