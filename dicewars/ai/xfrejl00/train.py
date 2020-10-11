import tensorflow as tf
import argparse
import gc
import os
import sys
import subprocess
import re
from datetime import datetime
from configparser import ConfigParser
from dicewars.ai.xfrejl00.qtable import QTable

class TrainExc(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Q-learning model from scratch or continue training from a snapshot.")
    parser.add_argument("--load_model", dest="load_model", action="store_true", default=False, help="Load weights and continue training.")
    parser.add_argument("--snapshot_path", dest="dest_folder", action="store", help="Choose from which folder the snapshot will be loaded. Defaults to the folder belonging the to latest training of model.")
    parser.add_argument("--matches_count", dest="matches_count", action="store", type=int, help="Number of matches the model will be trained for.")
    parser.add_argument("--snapshot_frequency", dest="snapshot_frequency", action="store", type=int, help="How many matches will be played before the model weights are saved.")
    parser.add_argument("--graphs_frequency", dest="graphs_frequency", action="store", type=int, help="How many matches will be played before the training graphs are saved.")
    parser.add_argument("--lr", dest="learning_rate", action="store", type=float, help="Learning rate, used in Bellman equation to define how quickly the model should be trained.")
    parser.add_argument("--discount", dest="discount", action="store", type=float, help="Also known as \"gamma\", used in Bellman equation to balance current and future rewards.")
    parser.add_argument("--epsilon", dest="epsilon", action="store", type=float, help="Chance to perform random move instead of greedy move based on Q-table value.")
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

    if os.path.exists(snapshot_path) == False or os.path.isfile(snapshot_path + "snapshot.pickle") == False:
        raise TrainExc("ERROR: Can't find file with saved snapshot.")

    return snapshot_path


def setup_config(path, learning_rate, epsilon, discount):
    config = ConfigParser()
    config.read("dicewars/ai/xfrejl00/config.ini")
    config.set("BASE", "SnapshotPath", path)
    config.set("BASE", "LearningRate", str(learning_rate))
    config.set("BASE", "Epsilon", str(epsilon))
    config.set("BASE", "Discount", str(discount))
    with open("dicewars/ai/xfrejl00/config.ini", "w") as configfile:
        config.write(configfile)


def train(matches_count=1000, 
        snapshot_frequency=100, 
        graphs_frequency=100, 
        snapshot_path=None,
        learning_rate=0.85, # Subject to change, will also decay during training
        epsilon=0.9, # Subject to change, will also decay during training
        discount=0.99): # Subject to change
    setup_config(snapshot_path, learning_rate, epsilon, discount)
    print("Snapshot path: " + snapshot_path)
    """
    TODO: Start the game against AI's
    TODO: Add snapshot saving
    TODO: Gather relevant data
        - Moving average of winrate against all anemies
        - Moving averate of winrate against AI that our AI was not trained on (for validation purposes)
        - Moving average of games with bad inputs (if we don't mitigate them by hardcoding some checks)
    """
    pass

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
