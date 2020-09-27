import copy
import json
from json.decoder import JSONDecodeError
import logging
import signal

from .timers import FischerTimer, FixedTimer


class TimeoutError(Exception):
    pass


def TimeoutHandler(signum, handler):
    raise TimeoutError('')


TIME_LIMIT_CONSTRUCTOR = 10.0  # in seconds, for AI constructor
FISCHER_INIT = 10.0  # seconds
FISCHER_INCREMENT = 0.25  # seconds


class BattleCommand:
    def __init__(self, source_name, target_name):
        self.source_name = source_name
        self.target_name = target_name


class EndTurnCommand:
    pass


class AIDriver:
    """Basic AI agent implementation
    """
    def __init__(self, game, ai_constructor):
        """
        Parameters
        ----------
        game : Game

        Attributes
        ----------
        board : Board
        waitingForResponse : bool
           Indicates whether agent is waiting for a response from the server
        """
        self.logger = logging.getLogger('AI')
        self.game = game
        self.board = game.board
        self.player_name = game.player_name

        signal.signal(signal.SIGALRM, TimeoutHandler)

        self.ai_disabled = False
        try:
            board_copy = copy.deepcopy(self.board)
            players_order_copy = copy.deepcopy(self.game.players_order)
            with FixedTimer(TIME_LIMIT_CONSTRUCTOR):
                self.ai = ai_constructor(self.player_name, board_copy, players_order_copy)
        except TimeoutError:
            self.logger.error("The AI failed to construct itself in {}s. Disabling it.".format(TIME_LIMIT_CONSTRUCTOR))
            self.ai_disabled = True
        except Exception:
            self.logger.error("The AI crashed during construction:\n", exc_info=True)
            self.ai_disabled = True

        self.waitingForResponse = False
        self.moves_this_turn = 0
        self.turns_finished = 0

        self.timer = FischerTimer(FISCHER_INIT, FISCHER_INCREMENT)

    def run(self):
        """Main AI agent loop
        """
        game = self.game

        while True:
            message = game.input_queue.get(block=True, timeout=None)
            try:
                if not self.handle_server_message(message):
                    exit(0)
            except JSONDecodeError:
                self.logger.error("Invalid message from server.")
                exit(1)
            self.current_player_name = game.current_player.get_name()
            if self.current_player_name == self.player_name and not self.waitingForResponse:
                if self.ai_disabled:
                    self.logger.warning("The AI has already misbehaved, just end-turning.")
                    self.send_message('end_turn')
                    continue

                try:
                    board_copy = copy.deepcopy(self.board)
                    with self.timer as time_left:
                        command = self.ai.ai_turn(
                            board_copy,
                            self.moves_this_turn,
                            self.turns_finished,
                            time_left
                        )
                    self.process_command(command)
                except TimeoutError:
                    self.logger.warning("Forced 'end_turn' because of timeout")
                    self.send_message('end_turn')
                    self.time_left_last_time = -1.0
                except Exception:
                    self.logger.error("The AI crashed during attempt to make a move:\n", exc_info=True)
                    self.send_message('end_turn')
                    self.ai_disabled = True

                if not self.waitingForResponse:
                    self.logger.warning("Forced 'end_turn' because the implementation did nothing")
                    self.send_message('end_turn')

    def handle_server_message(self, msg):
        """Process message from the server

        Parameters
        ----------
        msg : str
            Message from the server
        """
        self.logger.debug("Received message type {0}.".format(msg["type"]))
        if msg['type'] == 'battle':
            atk_data = msg['result']['atk']
            def_data = msg['result']['def']
            attacker = self.game.board.get_area(str(atk_data['name']))
            attacker.set_dice(atk_data['dice'])
            atk_name = attacker.get_owner_name()

            defender = self.game.board.get_area(def_data['name'])
            defender.set_dice(def_data['dice'])
            def_name = defender.get_owner_name()

            if def_data['owner'] == atk_data['owner']:
                defender.set_owner(atk_data['owner'])
                self.game.players[atk_name].set_score(msg['score'][str(atk_name)])
                self.game.players[def_name].set_score(msg['score'][str(def_name)])

            self.waitingForResponse = False

        elif msg['type'] == 'end_turn':
            current_player = self.game.players[self.game.current_player_name]

            for area in msg['areas']:
                owner_name = msg['areas'][area]['owner']

                area_object = self.game.board.get_area(int(area))

                area_object.set_owner(owner_name)
                area_object.set_dice(msg['areas'][area]['dice'])

            current_player.deactivate()
            self.game.current_player_name = msg['current_player']
            self.game.current_player = self.game.players[msg['current_player']]
            self.game.players[self.game.current_player_name].activate()
            self.waitingForResponse = False

        elif msg['type'] == 'game_end':
            self.logger.info("Player {} has won".format(msg['winner']))
            self.game.socket.close()
            return False

        return True

    def process_command(self, command):
        if isinstance(command, BattleCommand):
            if self.battle_is_valid(command):
                self.send_message('battle', command.source_name, command.target_name)
            else:
                self.send_message('end_turn')
        elif isinstance(command, EndTurnCommand):
            self.send_message('end_turn')
        else:
            raise RuntimeError("Unknown command: {}".format(command))

    def send_message(self, type, attacker=None, defender=None):
        """Send message to the server

        Parameters
        ----------
        type : str
        attacker : int
        defender : int
        """
        if type == 'battle':
            msg = {
                'type': 'battle',
                'atk': attacker,
                'def': defender
            }
            self.logger.debug("Sending battle message {}->{}".format(attacker, defender))
            self.moves_this_turn += 1
        elif type == 'end_turn':
            msg = {'type': 'end_turn'}
            self.logger.debug("Sending end_turn message.")
            self.moves_this_turn = 0
            self.turns_finished += 1
        else:
            raise RuntimeError("Attempt to send unexpected message type {}".format(type))

        self.waitingForResponse = True

        try:
            self.game.socket.send(str.encode(json.dumps(msg)))
        except BrokenPipeError:
            self.logger.error("Connection to server broken.")
            exit(1)

    def battle_is_valid(self, battle):
        try:
            source_area = self.board.get_area(battle.source_name)
        except KeyError:
            self.logger.error('Player {} specified area {} -- which is not even a valid area name!'.format(
                self.player_name, battle.source_name
            ))
            self.ai_disabled = True
            return False

        source_owner = source_area.get_owner_name()

        if source_owner != self.player_name:
            self.logger.error('Player {} attempted to attack from area {} owned by {}.'.format(
                self.player_name, battle.source_name, source_owner
            ))
            self.ai_disabled = True
            return False

        if not source_area.can_attack():
            self.logger.error('Attempted to attack from area {} having {} dice.'.format(
                battle.source_name, source_area.get_dice()
            ))
            self.ai_disabled = True
            return False

        if battle.target_name not in source_area.get_adjacent_areas():
            self.logger.error('Attempted to attack from area {} area {} which is not adjacent.'.format(
                battle.source_name, battle.target_name
            ))
            self.ai_disabled = True
            return False

        return True
