# Dice Wars

Dice Wars is a strategy game where players take turns to attack adjacent territories to expand
their area. Each territory contains a number of dice determining player's presence
and strength. The objective of the game is to conquer all territories and thus eliminate each opponent.

This is a client-server implementation that based on a [bachelor's thesis at FIT BUT](https://www.vutbr.cz/www_base/zav_prace_soubor_verejne.php?file_id=180901).


## Installation

To use this, you need to have python3 and the following python packages:

    hexutil
    numpy
    pyqt
    matplotlib

A standard ``requirements.txt`` is provided.

Furthermore, the root of the repository needs to be in ``PYTHONPATH``.

As an easy way of setting up the environment, do the following:

    # install
    mkdir SUI
    cd SUI
    git clone https://github.com/ibenes/dicewars.git repo
    cd repo
    bash install.sh

    # setup
    . path.sh
    # try a game
    python3 ./scripts/dicewars-human.py --ai dt.sdc dt.rand xlogin00 xlogin42


## Running the game

There are three different scripts prepared, which allow for testing different scenarios.
However, they all expose a common set of parameters for controlling the pseudo-randomness in the game:

    -b  geometry of the board
    -o  clustering of areas into possesion of individual players  
    -s  assignment of dice to areas

When not set, the source of pseudo-random numbers is seeded from current time, becoming effectively random.

Finally, individual AIs are referred to as follows:
For every ``module`` in ``dicewars.ai``, which contains a class ``AI``, the ``AI`` is identified by ``module``. Examples are given throughout the following sections.

### Playing with human
Starts a human-controlled client along those driven by AIs.
There can be between 1 and 7 AIs.
For an easy game (beware, defeat is still a real possibility), try:

    python3 ./scripts/dicewars-human.py --ai dt.sdc dt.rand xlogin00 xlogin42

For a really challenging setup, try:

    python3 ./scripts/dicewars-human.py --ai dt.sdc dt.ste dt.stei dt.wpm_c

### Playing with fixed AI order
Starts a set of games between AIs in given order.
Increments the board seed with every game.
Additionally exposes these options:

    -n      number of board to be played
    -l      folder where to put logs of last game
    -r      keep reporting which game is being played

An example:

    python3 ./scripts/dicewars-ai-only.py -r -b 11 -o 22 -s 33 -c 44 -n 10 -l ../logs --ai dt.stei xlogin42

### Running a tournament
Keeps picking a subset of AIs of specified size and has them play together.
The total set of AIs considered is given in the script itself.
Ownership (``-o``), dice assignment (``-s``), and dice rolls (``-f``) seeds are not exposed.
Additionally exposes these options:

    -n      number of boards to be played
    -g      size of games in number of players
    -l      folder where to put logs of last game
    -s      seed for selecting who plays whom
    -r      keep reporting what game is being played
    --save  where to save the resulting list of games

For every board, all rotations of a random permutation of the player order are played, thus the total number of games equals ``N x G``

An example:

    python3 ./scripts/dicewars-tournament.py -r -g 2 -n 50 -b 101 -s 1337 -l ../logs --save ../tournaments/tournament-g2-n50.pickle

This script can also be used for evaluation of a specific AI, ensuring that it takes part in every game played.
This is achieved through ``--ai-under-test``, e.g.:

    python3 ./scripts/dicewars-tournament.py -r -g 2 -n 50 --ai-under-test dt.sdc -b 101 -s 1337 -l ../logs

### Observing convergence of winrates
If you have saved games from a tournament (through its ``--save`` option), you can display the evolution of the winrates:

    python3 ./scripts/winrate-progress.py --xmin 10 ../tournaments/tournament-g2-n50.pickle 

Note that the evolution of winrates does not have any other interpretation than the rate of convergence!

## Implementing AIs
See ``dicewars/ai/template.py`` and other existing AIs in the package.
An AI is a class implementing two standard functions: ``__init__()`` and ``ai_turn()``


### Name vs. instance
Players and areas exist primary as instances of Player and Area.
However -- originally for serialization purposes -- they are both referred to by their "name".
These names are instances of `int`.
Board can return Areas as given by name and every Area knows its name.

There is no reason for an AI to access instances of Player.

## AI interface

The constructor is expected to take following parameters:

    player_name     the name of the player this AI will control
    board           an instance of dicewars.client.game.Board
    players_order   in what order do players take turns

The turn making method is expected to take following parameters:

    board               an instance of dicewars.client.game.Board   
    nb_moves_this_turn  number of attacks made in this turn
    nb_turns_this_game  number of turns ended so far
    previous_time_left  time (in seconds) left after last decision making

The ``AI.ai_turn()`` is required to return an instance of ``BattleCommand`` or ``EndTurnCommand``.

Multi-module implementation is possible, see ``xlogin42`` for an example.

## Learning about the world
Board's ``get_player_areas()``, ``get_player_border()``, and ``get_players_regions()`` can be used to discover areas belonging to any player in the game.
Instances of ``Area`` then allow further inquiry through ``get_adjacent_areas()``, ``get_owner_name()`` and ``get_dice()``.

It may also be practical to acquire all possible moves from ``dicewars.ai.utils.possible_attacks()``.
This module also provides formulas for probability of conquering and holding an Area.

The instance of ``Board`` passed to AI is a deepcopy, so the AI is free to mangle it in any way deemed useful.

### Debuging visually
In addition to whatever favourite debugging method you have, Dicewars provide a simplistic way of visually inspecting the state of the game.
There is `save_state()` function provided by `dicewars.ai.utils`, which creates a dump of the state which AI observes.

The saved state can than be loaded by `scripts/visual-debugger.py`.
This visual debugger allows displaying different information on areas (change label through the sole button in the interface) and a custom detailed information upon selection of an area of interest.

There is an example of saving games in `dicewars.ai.xlogin42.phased.AI`, and the visual debugger displays a bit of information this AI cares about.
It is expected that the developers of new AI will adjust the debugger's `DetailedAreaReporter` to the needs of their AI.

## Dealing with misbehaving AIs

* Slow AI -- AIs have a fixed 10s time for constructing themselves.
Additionally, a [Fischer clock](https://en.wikipedia.org/wiki/Time_control#Increment_and_delay_methods) of 10s and 0.25s increment limits time for decision making.
AI failing to make a decision will be stopped in deciding and a ``EndTurnCommand`` will be sent instead (but the increment is made anyway, so the AI will be able to continue playing).
AIs are informed about the time they have left through ``time_left``. 
* Stupid AI -- AI attempting to make an illegal move will be switched off, idling it for the rest of the game.
* Passive AI -- AI sending only ``EndTurnCommand`` will be quickly taken care of by other players. However, if no AI makes a move for 8 consecutive rounds, the game will be contumated and every player scores a defeat.
