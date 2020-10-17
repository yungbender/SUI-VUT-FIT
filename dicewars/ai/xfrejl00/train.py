import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gc
import os
import sys
import subprocess
import re
import random
import pickle
import pandas as pd
from datetime import datetime
from configparser import ConfigParser
from dicewars.ai.xfrejl00.qtable import QTable
from dicewars.ai.xfrejl00.utils import *

ai_list = ["dt.ste", "dt.sdc", "dt.stei", "dt.wpm_c", "dt.wpm_d", "dt.wpm_s", "xlogin00", "xlogin42"]

class TrainExc(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Q-learning model from scratch or continue training from a snapshot.")
    parser.add_argument("--load_model", dest="load_model", action="store_true", default=False, help="Load weights and continue training.")
    parser.add_argument("--snapshot_path", dest="dest_folder", action="store", help="Choose from which folder the snapshot will be loaded. Defaults to the folder belonging the to latest training of model.")
    parser.add_argument("--matches_count", dest="matches_count", action="store", type=int, help="Number of matches the model will be trained for.")
    parser.add_argument("--save_frequency", dest="save_frequency", action="store", type=int, help="How many matches will be played before the training graphs and match data are saved.")
    parser.add_argument("--lr", dest="learning_rate", action="store", type=float, help="Learning rate, used in Bellman equation to define how quickly the model should be trained.")
    parser.add_argument("--discount", dest="discount", action="store", type=float, help="Also known as \"gamma\", used in Bellman equation to balance current and future rewards.")
    parser.add_argument("--epsilon", dest="epsilon", action="store", type=float, help="Chance to perform random move instead of greedy move based on Q-table value.")
    parser.add_argument("--epsilon_decay", dest="epsilon_decay", action="store", type=float, help="Rate of epsilon decay. New_epsilon = Old_epsilon * epsilon_decay")
    parser.add_argument("--lr_decay", dest="learning_rate_decay", action="store", type=float, help="Rate of learning rate decay. New_lr = Old_lr * lr_decay")
    parser.add_argument("--min_lr", dest="min_learning_rate", action="store", type=float, help="Minimal value of learning rate during decaying.")
    parser.add_argument("--min_epsilon", dest="min_epsilon", action="store", type=float, help="Minimal value of epsilon during decaying.")

    return parser.parse_args()


def environment_setup():
    gc.enable()
    tf.keras.backend.clear_session()


def fetch_machine():
    # Print relevant info about machine
    print("Using computer: " + os.uname()[1])
    print("Current time: " + datetime.now().strftime('%Y.%m.%d %H:%M:%S'))
    print("Tensorflow version: " + tf.__version__)
    freeGpu = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
    if len(freeGpu) == 0:
        print('WARNING: No usable GPU available!')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = freeGpu.decode().strip()
        print("Found GPU: " + str(freeGpu))


def load_model(snapshot_path):
    if snapshot_path is None: # Get the snapshot from last training session
        dates = os.listdir('dicewars/ai/xfrejl00/snapshots/')

        r = re.compile("^\d{4}.\d{2}.\d{2} \d{2}:\d{2}:\d{2}$") # Filter out the custom non-date folders
        dates = list(filter(r.match, dates))

        dates.sort(reverse=True) # Sort by newest
        if len(dates) == 0:
            print("ERROR: Can't find any folder with saved snapshot.")
            sys.exit(1)
        snapshot_path = dates[0] # Get the newest folder

    snapshot_path = "dicewars/ai/xfrejl00/snapshots/" + snapshot_path + "/"

    if not os.path.exists(snapshot_path) or not os.path.isfile(snapshot_path + "snapshot.pickle"):
        raise TrainExc("ERROR: Can't find file with saved snapshot.")

    return snapshot_path


def setup_config(path):
    config = ConfigParser()
    config.read("dicewars/ai/xfrejl00/config.ini")
    config.set("BASE", "SnapshotPath", path)
    config.set("BASE", "Train", "True")
    with open("dicewars/ai/xfrejl00/config.ini", "w") as configfile:
        config.write(configfile)


def decay_lr_epsilon(lr, epsilon, lr_decay, epsilon_decay, min_lr, min_epsilon):
    lr = lr * lr_decay
    epsilon = epsilon * epsilon_decay
    return [max(lr, min_lr), max(epsilon, min_epsilon)]


def create_winrate_graphs(snapshot_path, df):
    # Calculate rolling average of winrate and save the stats to CSV file
    df["rolling_avg"] = df.iloc[:,0].rolling(window=200).mean()
    df.to_csv(snapshot_path + "winrate_all.csv", index=True)

    # Create graph
    plt.figure(figsize=[15,10])
    plt.grid(True)
    plt.plot(df["rolling_avg"], label="Win rate")
    plt.xlabel("Games played")
    plt.ylabel("Win rate")
    plt.legend(loc=2)
    plt.savefig(snapshot_path + "winrate_all.png")
    plt.close()

def train(matches_count=5000, 
        save_frequency=50, 
        snapshot_path=None,
        learning_rate=0.85, # Subject to change, will also decay during training
        epsilon=0.9, # Subject to change, will also decay during training
        discount=0.1, # Subject to change
        epsilon_decay=0.999, # Subject to change
        learning_rate_decay=0.999, # Subject to change
        min_learning_rate=0.3, # Subject to change
        min_epsilon=0.1, # Subject to change
        load_model=False,
        **kwargs):

    setup_config(snapshot_path)
    print("Snapshot path: " + snapshot_path)

    if load_model: # Load decayed parameters if continuing the run
        learning_rate, epsilon, discount = load_parameters(snapshot_path)
        df = pd.read_csv(snapshot_path + "winrate_all.csv", index_col=0)
    else: # Save the config and create csv file for stats if the run is starting for the first time
        save_parameters(snapshot_path, learning_rate, epsilon, discount)
        df = pd.DataFrame(columns=["win", "rolling_avg"], dtype=int)

    progress_bar = tf.keras.utils.Progbar(target=matches_count)
    q_table = QTable(states_count=3, action_count=1, qvalue_check=True)
    for i in range(matches_count):
        opponents = random.sample(ai_list , 3) + ["xfrejl00"] # Get 3 random opponents from list and add our AI
        random.shuffle(opponents) # Shuffle the list

        # Run and analyze the game
        game_output = subprocess.check_output(['python3', 'scripts/dicewars-ai-only.py', "--ai", opponents[0], opponents[1], opponents[2], opponents[3], "-d", "-l", "dicewars/logs"])
        won_game = bool(re.match(".*Winner: xfrejl00.*", game_output.decode("utf-8"))) # True - trained AI won, False - trained AI lost
        played_moves = load_moves_from_game(snapshot_path)

        # Calculate the reward
        reward = 0
        if won_game:
            reward += 10 * (2 + 200 / len(played_moves)) # Motivation to win ASAP
        else:
            placement = 4 - game_output.decode("utf-8").split(",").index("xfrejl00")

            reward += 10 - (5 * placement) # 2nd place = 0 reward, 3rd place = -5 reward, 4th place = -10 reward

        # Add the game info to pandas dataframe
        df = df.append({"win" : won_game}, ignore_index=True)

        # Create and save winrate graphs, snapshot backup 
        if i > 0 and i % save_frequency == 0:
            q_table.save(snapshot_path + "snapshot_backup.pickle")
            create_winrate_graphs(snapshot_path, df)

        q_table = q_table.load(snapshot_path + "snapshot.pickle")
        for move in played_moves: # Give reward to all played moves in last game
            # Bellman equation (for non-immediate rewards)
            q_table[move] = q_table[move] + learning_rate * reward

        # Decay learning rate and epsilon
        learning_rate, epsilon = decay_lr_epsilon(learning_rate, epsilon, learning_rate_decay, epsilon_decay, min_learning_rate, min_epsilon)

        # Save the Q-table and training config
        q_table.save(snapshot_path + "snapshot.pickle")
        save_parameters(snapshot_path, learning_rate, epsilon, discount)
    
        progress_bar.update(i+1)
    """
    TODO: Gather relevant data
        - Moving averate of winrate against AI that our AI was not trained on (for validation purposes)
    """

def main():
    args = parse_args()
    environment_setup()
    fetch_machine()

    if args.load_model: # Load selected or latest snapshot
        path = load_model(args.dest_folder)
    else: # New snapshot
        if args.dest_folder: # New snapshot with selected name
            path = "dicewars/ai/xfrejl00/snapshots/" + args.dest_folder + "/"
        else: # New snapshot with default name
            path = "dicewars/ai/xfrejl00/snapshots/" + datetime.now().strftime('%Y.%m.%d %H:%M:%S') + "/"
        os.makedirs(path)

        q_table = QTable() # Create new Q-table
        q_table.save(path + "snapshot.pickle")

    args = dict((arg, getattr(args,arg)) for arg in vars(args) if getattr(args, arg) is not None) # Remove not empty args so we can get default values of arguments
    train(**args, snapshot_path=path)

if __name__ == "__main__":
    main()
