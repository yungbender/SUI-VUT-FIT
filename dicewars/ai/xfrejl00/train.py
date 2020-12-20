import matplotlib.pyplot as plt
import numpy as np
import argparse
import collections
import gc
import os
import sys
import subprocess
import re
import random
import pickle
import pandas as pd
import signal
import torch
import copy
from tqdm import tqdm
from datetime import datetime
from configparser import ConfigParser
from dicewars.ai.xfrejl00.qtable import QTable
from dicewars.ai.xfrejl00.utils import *
from scripts.utils import run_ai_only_game, BoardDefinition
from dicewars.ai.xfrejl00.classifier import LogisticRegressionMultiFeature as Classifier
from dicewars.ai.xfrejl00.dqn import LogisticRegressionMultiFeature as DQNetwork

ai_list = ["dt.ste", "dt.sdc", "dt.stei", "dt.wpm_c", "dt.wpm_d", "dt.wpm_s", "xlogin00", "xlogin42", "nop", "alphadice-1", "alphadice-2"]
ai_val = ["dt.rand", "dt.ste", "dt.sdc", "dt.wpm_c", "xlogin00"]
SIGINT_CALLED = False

REPLAY_MEMORY_MAX = 100000
REPLAY_MEMORY_MIN = 4096
NB_FEATURES_CLASSIFIER = 22 # Number of classifier features
NB_FEATURES_DQN = 11 # 6 states, board states, custom stats
USE_DQN = False
BATCH_SIZE = 2048
LEARNING_RATE = 0.001

class TrainExc(Exception):
    pass


def terminate(*_):
    print("Requesting finish...")
    global SIGINT_CALLED
    SIGINT_CALLED = True


def children_ignore():
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    for sig in signals:
        signal.signal(sig, signal.SIG_IGN)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Q-learning model from scratch or continue training from a snapshot.")
    parser.add_argument("--evaluate", dest="evaluate", action="store_true", default=False, help="Run the AI against same AIs that it will encounter during grading.")
    parser.add_argument("--load_model", dest="load_model", action="store_true", default=False, help="Load weights and continue training.")
    parser.add_argument("--snapshot_path", dest="dest_folder", action="store", help="Choose from which folder the snapshot will be loaded. Defaults to the folder belonging the to latest training of model.")
    parser.add_argument("--match_count", dest="match_count", action="store", type=int, help="Number of matches the model will be trained for.")
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


def decay_epsilon(epsilon, epsilon_decay, min_epsilon):
    epsilon = epsilon * epsilon_decay
    return max(epsilon, min_epsilon)

def save_snapshots(snapshot_path, q_table, df, name, save=True):
    # Calculate rolling average of winrate and save the stats to CSV file
    max_winrate = df.iloc[:,1].max()
    df["rolling_avg"] = df.iloc[:,0].rolling(window=200).mean()
    df["rolling_avg_2000"] = df.iloc[:,0].rolling(window=2000).mean()
    df.to_csv(snapshot_path + name + ".csv", index=True)

    if save: # During training, we save snapshot backups and best models
        q_table.save(snapshot_path + "snapshot_backup.pickle", deepcopy=True)
        
        new_max_winrate = df.iloc[:,1].max()
        if new_max_winrate >= max_winrate: # New biggest winrate, save the snapshot to separate folder
            os.makedirs(snapshot_path + "records/", exist_ok=True)
            q_table.save(snapshot_path + "records/" + name + "_" + str(new_max_winrate) + ".pickle", deepcopy=True)

    return df

def create_training_graphs(snapshot_path, losses_dqn, accuracies_dqn, losses_classifier, accuracies_classifier):
    x = np.linspace(1, len(losses_classifier), len(losses_classifier))
    plt.ylabel("Loss value")
    plt.xlabel("Epochs")
    plt.plot(x, losses_classifier, color="blue")
    plt.legend(["Training loss"])
    plt.savefig(snapshot_path + "classifier_losses" + ".png")
    plt.close()

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(x, accuracies_classifier, color="blue")
    plt.legend(["Validation accuracy"])
    plt.savefig(snapshot_path + "classifier_accuracy" + ".png")
    plt.close()

    x = np.linspace(1, len(losses_dqn), len(losses_dqn))
    plt.ylabel("Loss value")
    plt.xlabel("Epochs")
    plt.plot(x, losses_dqn, color="blue")
    plt.legend(["Training loss"])
    plt.savefig(snapshot_path + "dqn_losses" + ".png")
    plt.close()

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(x, accuracies_dqn, color="blue")
    plt.legend(["Validation accuracy"])
    plt.savefig(snapshot_path + "dqn_accuracy" + ".png")
    plt.close()

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

def initialize_buffer(positives, negatives):
    label_buffer = collections.deque(maxlen=REPLAY_MEMORY_MAX)
    statistics_buffer = collections.deque(maxlen=REPLAY_MEMORY_MAX)

    with open(positives, "r") as positives_file, open(negatives, "r") as negatives_file:
        num_positives = sum(1 for line in positives_file)
        num_negatives = sum(1 for line in negatives_file)

    # We do this to keep the share of positives/negatives the same
    positives_count = num_positives / (num_negatives + num_negatives) * REPLAY_MEMORY_MAX
    negatives_count = num_negatives / (num_negatives + num_negatives) * REPLAY_MEMORY_MAX

    positives = np.genfromtxt(positives, skip_header=max(int(num_positives - positives_count), 0), dtype=float)
    negatives = np.genfromtxt(negatives, skip_header=max(int(num_negatives - negatives_count), 0), dtype=float)

    for line in positives:
        statistics_buffer.append(line)
        label_buffer.append(1)

    for line in negatives:
        statistics_buffer.append(line)
        label_buffer.append(0)


    return statistics_buffer, label_buffer

def run_game(ai_list):
    process_list = [] # If we would like to handle signals
    board = BoardDefinition(None, None, None)
    result = run_ai_only_game(5005, "127.0.0.1", process_list, ai_list, board, logdir="dicewars/logs", debug=True)
    return result

def evaluate(match_count=1000, save_frequency=50, snapshot_path=None, **kwargs):
    if snapshot_path is None:
        print("Can't evaluate when there's no model given.")
        exit(-1)

    setup_config(snapshot_path, train=False)
    print("Snapshot path: " + snapshot_path)
    progress_bar = tqdm(total=match_count)
    df = pd.DataFrame(columns=["win", "rolling_avg"], dtype=int)

    for i in range(0, match_count, 4):
        opponents = random.sample(ai_val, 3) + ["xfrejl00"] # Get 3 random opponents from list and add our AI
        random.shuffle(opponents) # Shuffle the list
        for j in range(4):
            # Run and analyze the game
            game_output = run_game(opponents)
            opponents = np.roll(opponents, 1) # Rotate the opponents list
            won_game = bool(re.search(".*Winner: xfrejl00.*", game_output)) # True - trained AI won, False - trained AI lost
            with open("dicewars/logs/client-xfrejl00.log", "r") as f:
                if re.search(".*Traceback.*", f.read()):
                    print("Error: AI crashed during game.")
                    exit(-1)

            # Add the game info to pandas dataframe
            df = df.append({"win" : won_game}, ignore_index=True)
            if i > 0 and (i + j) % save_frequency == 0:
                df = save_snapshots(snapshot_path, None, df, "val_winrate", save=False)
                create_winrate_graphs(snapshot_path, df["rolling_avg"], "val_winrate")
                create_winrate_graphs(snapshot_path, df["rolling_avg_2000"], "val_winrate_2000")
            
            progress_bar.update(1)
            if SIGINT_CALLED:
                return

def train(match_count=5000, 
        save_frequency=50, 
        snapshot_path=None,
        learning_rate=1,
        epsilon=0.9,
        discount=0.999,
        epsilon_decay=0.999,
        learning_rate_decay=1,
        min_learning_rate=0.1,
        min_epsilon=0.05,
        load_model=False,
        **kwargs):

    setup_config(snapshot_path)
    print("Snapshot path: " + snapshot_path)

    # Initialize replay buffers
    classifier_label_buffer = collections.deque(maxlen=REPLAY_MEMORY_MAX)
    classifier_statistics_buffer = collections.deque(maxlen=REPLAY_MEMORY_MAX)
    dqn_label_buffer = collections.deque(maxlen=REPLAY_MEMORY_MAX)
    dqn_statistics_buffer = collections.deque(maxlen=REPLAY_MEMORY_MAX)
    dqn = DQNetwork(NB_FEATURES_DQN, LEARNING_RATE)
    classifier = Classifier(NB_FEATURES_CLASSIFIER, LEARNING_RATE)
    losses_classifier = []
    accuracies_classifier = []
    losses_dqn = []
    accuracies_dqn = []

    if load_model: # Load decayed parameters if continuing the run
        learning_rate, epsilon, discount = load_parameters(snapshot_path)
        df = pd.read_csv(snapshot_path + "winrate_all.csv", index_col=0)

        # Load classifier and DQN models
        classifier.load_state_dict(torch.load(snapshot_path + f"classifier_model_{NB_FEATURES_CLASSIFIER}.pt"))
        dqn.load_state_dict(torch.load(snapshot_path + f"dqn_model_{NB_FEATURES_DQN}.pt"))

        # Initialize buffers
        dqn_statistics_buffer, dqn_label_buffer = initialize_buffer(snapshot_path + "dqn/positives.txt", snapshot_path + "dqn/negatives.txt")
        classifier_statistics_buffer, classifier_label_buffer = initialize_buffer(snapshot_path + "classifier/positives.txt", snapshot_path + "classifier/negatives.txt")

        start_index = len(df.iloc[:])
        if start_index == match_count:
            print("More games have already been played during previous runs, raise your match_count.")
    else: # Save the config and create csv file for stats if the run is starting for the first time
        save_parameters(snapshot_path, learning_rate, epsilon, discount)

        # Create empty neural network models
        torch.save(classifier.state_dict(), snapshot_path + f"classifier_model_{NB_FEATURES_CLASSIFIER}.pt")
        torch.save(dqn.state_dict(), snapshot_path + f"dqn_model_{NB_FEATURES_DQN}.pt")

        df = pd.DataFrame(columns=["win", "rolling_avg"], dtype=int)
        start_index = 0

    progress_bar = tqdm(total=match_count, initial=start_index)

    for i in range(start_index, match_count, 4):
        opponents = random.sample(ai_list, 3) + ["xfrejl00"] # Get 3 random opponents from list and add our AI
        random.shuffle(opponents) # Shuffle the list
        for j in range(4):
            # Clear the statistics file from classifier and raw DQN labels file
            with open(snapshot_path + "classifier/statistics.txt", "w"), open(snapshot_path + "dqn/statistics_raw.txt", "w"):
                pass 
            
            # Run and analyze the game
            game_output = run_game(opponents)
            opponents = np.roll(opponents, 1) # Rotate the opponents list
            won_game = bool(re.search(".*Winner: xfrejl00.*", game_output)) # True - trained AI won, False - trained AI lost
            played_moves = load_moves_from_game(snapshot_path)
            with open("dicewars/logs/client-xfrejl00.log", "r") as f:
                if re.search(".*Traceback.*", f.read()):
                    print("Error: AI crashed during game.")
                    exit(-1)

            # Calculate the reward
            reward = 0
            inactivity_loss = False
            if won_game:
                bonus = (2 + 300 / len(played_moves))
                reward += 20 * bonus # Motivation to win ASAP
            else:
                placement = 4 - game_output.split(",").index("xfrejl00")

                bonus = -placement
                reward += 10 * bonus # 2nd place = -20 reward, 3rd place = -30 reward, 4th place = -40 reward

                with open("dicewars/logs/server.txt", "r") as f: # If game ended because AI decided not to play, we give it huge negative reward
                    if re.search("INFO:SERVER:Game cancelled because the limit of.*", f.read()):
                        bonus = reward = -200
                        inactivity_loss = True
                        print("Game cancelled because the AI decided not to play.")

            # Move the moves from game to correct dataset files 
            with open(snapshot_path + "classifier/statistics.txt", "r") as stats_classifier, open(snapshot_path + "dqn/statistics_raw.txt", "r") as raw_statistics_dqn:
                if won_game:
                    with open(snapshot_path + "classifier/positives.txt", "a+") as dataset_classifier:
                        for line in stats_classifier:
                            dataset_classifier.write(line)
                            classifier_statistics_buffer.append(np.fromstring(line, sep=" ", dtype=float))
                            classifier_label_buffer.append(1)

                    with open(snapshot_path + "dqn/positives.txt", "a+") as positives_dqn:
                        for line in raw_statistics_dqn:
                            #line = line.split() # Convert to list
                            #line[-1] = str(float(line[-1]) + bonus) # Add after-game bonus to reward
                            #line = " ".join(line) # Convert back to string
                            #positives_dqn.write(line + "\n")
                            positives_dqn.write(line)
                            dqn_statistics_buffer.append(np.fromstring(line, sep=" ", dtype=float))
                            dqn_label_buffer.append(1)
                else:
                    with open(snapshot_path + "classifier/negatives.txt", "a+") as dataset_classifier:
                        for line in stats_classifier:
                            dataset_classifier.write(line)
                            classifier_statistics_buffer.append(np.fromstring(line, sep=" ", dtype=float))
                            classifier_label_buffer.append(0)

                    with open(snapshot_path + "dqn/negatives.txt", "a+") as negatives_dqn:
                        for line in raw_statistics_dqn:
                            #line = line.split() # Convert to list
                            #line[-1] = str(float(line[-1]) + bonus) # Add after-game bonus to reward
                            #line = " ".join(line) # Convert back to string
                            #negatives_dqn.write(line + "\n")
                            negatives_dqn.write(line)
                            dqn_statistics_buffer.append(np.fromstring(line, sep=" ", dtype=float))
                            dqn_label_buffer.append(0)
            
            # Buffer lengths MUST match
            assert(len(dqn_label_buffer) == len(dqn_statistics_buffer))
            assert(len(classifier_label_buffer) == len(classifier_statistics_buffer))

            if (i + j) % save_frequency == 0 and i > 0:
                for _ in range(save_frequency):
                    # Train DQN
                    loss_epoch = 0
                    if len(dqn_label_buffer) > REPLAY_MEMORY_MIN and len(dqn_statistics_buffer) > REPLAY_MEMORY_MIN:
                        for x, t in dqn.batch_provider(np.array(dqn_statistics_buffer, dtype=float), np.array(dqn_label_buffer, dtype=float), BATCH_SIZE):
                            loss_epoch += dqn.training_step(x, t)
                        
                        losses_dqn.append(loss_epoch / (int(len(dqn_statistics_buffer) / BATCH_SIZE)))
                        accuracies_dqn.append(dqn.evaluate(np.array(dqn_statistics_buffer, dtype=float), np.array(dqn_label_buffer, dtype=float)))

                    # Train classifier
                    loss_epoch = 0
                    if len(classifier_statistics_buffer) > REPLAY_MEMORY_MIN and len(classifier_label_buffer) > REPLAY_MEMORY_MIN:
                        for x, t in classifier.batch_provider(np.array(classifier_statistics_buffer, dtype=float), np.array(classifier_label_buffer, dtype=float), BATCH_SIZE):
                            loss_epoch += classifier.training_step(x, t)
                        
                        losses_classifier.append(loss_epoch / (int(len(classifier_statistics_buffer) / BATCH_SIZE)))
                        accuracies_classifier.append(classifier.evaluate(np.array(classifier_statistics_buffer, dtype=float), np.array(classifier_label_buffer, dtype=float)))

                classifier_save = copy.deepcopy(classifier) 
                torch.save(classifier_save.state_dict(), snapshot_path + f"classifier_model_{NB_FEATURES_CLASSIFIER}.pt")
                dqn_save = copy.deepcopy(dqn)
                torch.save(dqn_save.state_dict(), snapshot_path + f"dqn_model_{NB_FEATURES_DQN}.pt")

            # Add the game info to pandas dataframe
            df = df.append({"win" : won_game}, ignore_index=True)

            q_table = QTable.load(snapshot_path + "snapshot.pickle")
            for move in played_moves: # Give reward to all played moves in last game
                # Bellman equation (for non-immediate rewards)
                q_table[move] = q_table[move] * discount #+ learning_rate * reward
                q_table = give_reward_to_better_turns(q_table, reward, learning_rate, move, 2, ["very low", "low", "medium", "high"])
                reward *= 1 - 1 / len(played_moves) # Value end-game moves more than early game moves

            # Create and save winrate graphs, snapshot backup
            if i > 0 and (i + j) % save_frequency == 0:
                df = save_snapshots(snapshot_path, q_table, df, "winrate_all")
                create_training_graphs(snapshot_path, losses_dqn, accuracies_dqn, losses_classifier, accuracies_classifier)
                create_winrate_graphs(snapshot_path, df["rolling_avg"], "winrate_all")
                create_winrate_graphs(snapshot_path, df["rolling_avg_2000"], "winrate_all_2000")

            # Decay epsilon
            epsilon = decay_epsilon(epsilon, epsilon_decay, min_epsilon)

            # Save the Q-table and training config
            q_table.save(snapshot_path + "snapshot.pickle")
            q_table.close()
            save_parameters(snapshot_path, learning_rate, epsilon, discount)

            progress_bar.update(1)

            if SIGINT_CALLED:
                return


def main():
    args = parse_args()
    environment_setup()
    fetch_machine()

    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    for sig in signals:
        signal.signal(sig, terminate)

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
        os.makedirs(path + "classifier/", exist_ok=True)
        os.makedirs(path + "dqn/", exist_ok=True)

        q_table = QTable() # Create new Q-table
        q_table.save(path + "snapshot.pickle")
        q_table.close()

    args_cleaned = dict((arg, getattr(args,arg)) for arg in vars(args) if getattr(args, arg) is not None) # Remove not empty args so we can get default values of arguments
    if args.evaluate:
        evaluate(**args_cleaned, snapshot_path=path)
    else:
        train(**args_cleaned, snapshot_path=path)

if __name__ == "__main__":
    main()
