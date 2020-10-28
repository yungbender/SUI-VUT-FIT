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
from tqdm import tqdm
from datetime import datetime
from configparser import ConfigParser
from dicewars.ai.xfrejl00.qtable import QTable
from dicewars.ai.xfrejl00.utils import *

ai_list = ["dt.ste", "dt.sdc", "dt.stei", "dt.wpm_c", "dt.wpm_d", "dt.wpm_s", "xlogin00", "xlogin42", "nop", "alphadice-1"]
ai_val = ["dt.rand", "dt.ste", "dt.sdc", "dt.wpm_c", "xlogin00"]

class TrainExc(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Q-learning model from scratch or continue training from a snapshot.")
    parser.add_argument("--evaluate", dest="evaluate", action="store_true", default=False, help="Run the AI against same AIs that it will encounter during grading.")
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
    #tf.keras.backend.clear_session()


def fetch_machine():
    # Print relevant info about machine
    print("Using computer: " + os.uname()[1])
    print("Current time: " + datetime.now().strftime('%Y.%m.%d %H:%M:%S'))
    #print("Tensorflow version: " + tf.__version__)
    #freeGpu = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
    #if len(freeGpu) == 0:
    #    print('WARNING: No usable GPU available!')
    #else:
    #    os.environ['CUDA_VISIBLE_DEVICES'] = freeGpu.decode().strip()
    #    print("Found GPU: " + str(freeGpu))


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


def setup_config(path, train=True):
    config = ConfigParser()
    config.read("dicewars/ai/xfrejl00/config.ini")
    config.set("BASE", "SnapshotPath", path)
    config.set("BASE", "Train", str(train))
    with open("dicewars/ai/xfrejl00/config.ini", "w") as configfile:
        config.write(configfile)


def decay_lr_epsilon(lr, epsilon, lr_decay, epsilon_decay, min_lr, min_epsilon):
    lr = lr * lr_decay
    epsilon = epsilon * epsilon_decay
    return [max(lr, min_lr), max(epsilon, min_epsilon)]

def save_snapshots(snapshot_path, q_table, df, name, save=True):
    q_table.save(snapshot_path + "snapshot_backup.pickle")
    # Calculate rolling average of winrate and save the stats to CSV file
    max_winrate = df.iloc[:,1].max()
    df["rolling_avg"] = df.iloc[:,0].rolling(window=200).mean()
    df["rolling_avg_2000"] = df.iloc[:,0].rolling(window=2000).mean()
    if save:
        df.to_csv(snapshot_path + name + ".csv", index=True)
        
        new_max_winrate = df.iloc[:,1].max()
        if max_winrate < new_max_winrate: # New biggest winrate, save the snapshot to separate folder
            os.makedirs(snapshot_path + "records/", exist_ok=True)
            q_table.save(snapshot_path + "records/" + name + "_" + str(new_max_winrate) + ".pickle")

    return df

def create_winrate_graphs(snapshot_path, data, name):
    # Create graphs
    plt.figure(figsize=[15,10])
    plt.grid(True)
    plt.plot(data, label="Win rate")
    plt.xlabel("Games played")
    plt.ylabel("Win rate")
    plt.legend(loc=2)
    plt.savefig(snapshot_path + name + ".png")
    plt.close()

def evaluate(matches_count=1000, save_frequency=50, snapshot_path=None, **kwargs):
    if snapshot_path is None:
        print("Can't evaluate when there's no model given.")
        exit(-1)

    setup_config(snapshot_path, train=False)
    print("Snapshot path: " + snapshot_path)
    progress_bar = tqdm(total=matches_count)
    df = pd.DataFrame(columns=["win", "rolling_avg"], dtype=int)

    for i in range(0, matches_count, 4):
        opponents = random.sample(ai_val, 3) + ["xfrejl00"] # Get 3 random opponents from list and add our AI
        random.shuffle(opponents) # Shuffle the list
        for j in range(4):
            # Run and analyze the game
            game_output = subprocess.check_output(['python3', 'scripts/dicewars-ai-only.py', "--ai", opponents[0], opponents[1], opponents[2], opponents[3], "-d", "-l", "dicewars/logs"])
            opponents = np.roll(opponents, 1) # Rotate the opponents list
            won_game = bool(re.search(".*Winner: xfrejl00.*", game_output.decode("utf-8"))) # True - trained AI won, False - trained AI lost
            with open("dicewars/logs/client-xfrejl00.log", "r") as f:
                if re.search(".*Traceback.*", f.read()):
                    print("Error: AI crashed during game.")
                    exit(-1)

            # Add the game info to pandas dataframe
            df = df.append({"win" : won_game}, ignore_index=True)
            if i > 0 and (i + j) % save_frequency == 0:
                df = save_snapshots(snapshot_path, q_table, df, "val_winrate", save=True)
                create_winrate_graphs(snapshot_path, df["rolling_avg"], "val_winrate")
                create_winrate_graphs(snapshot_path, df["rolling_avg_2000"], "val_winrate_2000")
            
            progress_bar.update(1)

def train(matches_count=5000, 
        save_frequency=50, 
        snapshot_path=None,
        learning_rate=0.85, # Subject to change, will also decay during training
        epsilon=0.9, # Subject to change, will also decay during training
        discount=0.99, # Subject to change
        epsilon_decay=0.999, # Subject to change
        learning_rate_decay=0.9995, # Subject to change
        min_learning_rate=0.001, # Subject to change
        min_epsilon=0, # Subject to change
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

    progress_bar = tqdm(total=matches_count)

    for i in range(0, matches_count, 4):
        opponents = random.sample(ai_list , 3) + ["xfrejl00"] # Get 3 random opponents from list and add our AI
        random.shuffle(opponents) # Shuffle the list
        for j in range(4):
            # Run and analyze the game
            game_output = subprocess.check_output(['python3', 'scripts/dicewars-ai-only.py', "--ai", opponents[0], opponents[1], opponents[2], opponents[3], "-d", "-l", "dicewars/logs"])
            opponents = np.roll(opponents, 1) # Rotate the opponents list
            won_game = bool(re.search(".*Winner: xfrejl00.*", game_output.decode("utf-8"))) # True - trained AI won, False - trained AI lost
            played_moves = load_moves_from_game(snapshot_path)
            with open("dicewars/logs/client-xfrejl00.log", "r") as f:
                if re.search(".*Traceback.*", f.read()):
                    print("Error: AI crashed during game.")
                    exit(-1)

            # Calculate the reward
            reward = 0
            if won_game:
                reward += 10 * (2 + 200 / len(played_moves)) # Motivation to win ASAP
            else:
                placement = 4 - game_output.decode("utf-8").split(",").index("xfrejl00")

                reward += 11 - (5 * placement) # 2nd place = -1 reward, 3rd place = -6 reward, 4th place = -11 reward

            # Add the game info to pandas dataframe
            df = df.append({"win" : won_game}, ignore_index=True)

            # Create and save winrate graphs, snapshot backup
            if i > 0 and (i + j) % save_frequency == 0:
                df = save_snapshots(snapshot_path, q_table, df, "winrate_all")
                create_winrate_graphs(snapshot_path, df["rolling_avg"], "winrate_all")
                create_winrate_graphs(snapshot_path, df["rolling_avg_2000"], "winrate_all_2000")

            q_table = QTable.load(snapshot_path + "snapshot.pickle")
            for move in played_moves: # Give reward to all played moves in last game
                # Bellman equation (for non-immediate rewards)
                q_table[move] = q_table[move] * discount + learning_rate * reward
                q_table = give_reward_to_better_turns(q_table, reward, learning_rate, move, 2, ["very low", "low", "medium", "high"])
                q_table = give_reward_to_better_turns(q_table, reward, learning_rate, move, 3, ["very low", "low", "medium", "high"])

            # Decay learning rate and epsilon
            learning_rate, epsilon = decay_lr_epsilon(learning_rate, epsilon, learning_rate_decay, epsilon_decay, min_learning_rate, min_epsilon)

            # Save the Q-table and training config
            q_table.save(snapshot_path + "snapshot.pickle")
            save_parameters(snapshot_path, learning_rate, epsilon, discount)

            progress_bar.update(1)

def main():
    args = parse_args()
    environment_setup()
    fetch_machine()

    if args.load_model or args.evaluate: # Load selected or latest snapshot
        path = load_model(args.dest_folder)
    else: # New snapshot
        if args.dest_folder: # New snapshot with selected name
            path = "dicewars/ai/xfrejl00/snapshots/" + args.dest_folder + "/"
            if os.path.exists(path) and os.path.isfile(path + "snapshot.pickle") and not args.evaluate:
                print("Error: Snapshot with this name already exists. Delete it, change snapshot name or start again with --load_model.")
                exit(-1)
        else: # New snapshot with default name
            path = "dicewars/ai/xfrejl00/snapshots/" + datetime.now().strftime('%Y.%m.%d %H:%M:%S') + "/"
        os.makedirs(path, exist_ok=True)

        q_table = QTable() # Create new Q-table
        q_table.save(path + "snapshot.pickle")

    args_cleaned = dict((arg, getattr(args,arg)) for arg in vars(args) if getattr(args, arg) is not None) # Remove not empty args so we can get default values of arguments
    if args.evaluate:
        evaluate(**args_cleaned, snapshot_path=path)
    else:
        train(**args_cleaned, snapshot_path=path)

if __name__ == "__main__":
    main()
