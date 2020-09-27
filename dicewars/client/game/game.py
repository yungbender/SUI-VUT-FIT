import json
from json.decoder import JSONDecodeError
import logging
import socket
from queue import Queue
from time import sleep

from .board import Board
from .player import Player
from dicewars.client.socket_listener import SocketListener


class Game:
    """Represantation of the game state
    """
    def __init__(self, addr, port, hello_msg):
        """
        Parameters
        ----------
        addr : str
            Server address
        port : int
            Server port
        """
        self.logger = logging.getLogger('CLIENT')

        self.buffer = 65535
        self.battle_in_progress = False

        self.server_address = addr
        self.server_port = port
        self.players = {}

        i = 0
        while True:
            try:
                self.init_socket()
                break
            except ConnectionRefusedError as e:
                if i > 100:
                    self.logger.error("Connection to server refused: {0}".format(e))
                    exit(1)
                i += 1
                sleep(0.01)

        try:
            self.socket.send(str.encode(json.dumps(hello_msg)))
        except BrokenPipeError:
            self.logger.error("Connection to server broken.")
            exit(1)

        self.start_socket_daemon()
        while self.input_queue.empty():
            pass
        msg = self.input_queue.get()

        self.logger.debug("Received message: {0}\n".format(msg))  # TODO
        if msg['type'] == 'game_start':
            self.player_name = msg['player']
            self.add_players(int(msg['no_players']), msg['score'])
            self.board = Board(msg['areas'], msg['board'])
            self.current_player = self.players[msg['current_player']]
            self.current_player_name = msg['current_player']
            self.players_order = msg['order']
        else:
            self.logger.error("Did not receive game state from server.")
            exit(1)

        self.logger.info("This is player name {}, the players order is {}".format(self.player_name, self.players_order))

    ##################
    # INITIALIZATION #
    ##################
    def add_players(self, number_of_players, score):
        """Create Players instances
        
        Parameters
        ----------
        number_of_players : int
        score : list of int
            Initial scores of all players
        """
        self.number_of_players = number_of_players

        for i in range(1, number_of_players + 1):
            self.players[i] = Player(i, score[str(i)])

    ##############
    # NETWORKING #
    ##############
    def send_message(self, type, attacker=None, defender=None):
        """Send message to the server

        Parameters
        ----------
        type : str
        attacker : int
            Name of attacking area
        defender : int
            Name of defending area
        """
        if type == 'close':
            msg = {'type': 'close'}
        elif type == 'battle':
            msg = {
                'type': 'battle',
                'atk': attacker,
                'def': defender
            }
        elif type == 'end_turn':
            msg = {'type': 'end_turn'}
            self.logger.debug("Sending end_turn message.")

        try:
            self.socket.send(str.encode(json.dumps(msg)))
        except BrokenPipeError:
            self.logger.error("Connection to server broken.")
            exit(1)

    def init_socket(self):
        """Socket initialization
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.server_address, self.server_port))

    def start_socket_daemon(self):
        """Start message collecting daemon
        """
        self.input_queue = Queue()
        self.socket_listener = SocketListener(self.socket, self.buffer, self.input_queue)
        self.socket_listener.daemon = True
        self.socket_listener.start()
        self.logger.debug("Started socket daemon.")
