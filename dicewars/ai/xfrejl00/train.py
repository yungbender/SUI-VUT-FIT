import tensorflow as tf
import argparse
import gc
import os
import sys
import subprocess
import re
from datetime import datetime
from model import AlphaDice

parser = argparse.ArgumentParser(description="Train a Q-learning model from scratch or continue training from a snapshot.")
parser.add_argument("--load_model", dest="load_model", action="store_true", default=False, help="Load weights and continue training.")
parser.add_argument("--snapshot_path", dest="snapshot_path", action="store", help="Choose from which folder the snapshot will be loaded. Defaults to the folder belonging the to latest training of model.")
parser.add_argument("--matches_count", dest="matches_count", action="store", type=int, help="Number of matches the model will be trained for.")
parser.add_argument("--snapshot_frequency", dest="snapshot_frequency", action="store", type=int, help="How many matches will be played before the model weights are saved.")
parser.add_argument("--graphs_frequency", dest="graphs_frequency", action="store", type=int, help="How many matches will be played before the training graphs are saved.")
args = parser.parse_args()

# Save the args into dictionary
init_args = {} 
for arg in vars(args):
    if getattr(args, arg) is not None and arg not in ["load_model", "snapshot_path"]:
        init_args[arg] = getattr(args, arg)

# Force memory freeing
gc.enable()
tf.keras.backend.clear_session()

# Print relevant info about machine
print("Using computer: " + os.uname()[1])
print("Current time: " + datetime.now().strftime('%Y.%m.%d %H:%M:%S'))
print("Tensorflow version = " + tf.__version__)
freeGpu = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
if len(freeGpu) == 0:
    print('WARNING: No usable GPU available!')
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = freeGpu.decode().strip()
    print("Found GPU: " + str(freeGpu))

model = AlphaDice(**init_args)

# Load the model weights if needed
if args.load_model:
    if args.snapshot_path is None: # Get the snapshot from last training session
        dates = os.listdir('dicewars/ai/xfrejl00/snapshots/')

        r = re.compile("^\d{4}.\d{2}.\d{2} \d{2}:\d{2}:\d{2}$") # Filter out the custom non-date folders
        dates = list(filter(r.match, dates))

        dates.sort(reverse=True) # Sort by newest
        if len(dates) == 0:
            print("ERROR: Can't find any folder with saved weights.")
            sys.exit(1)
        args.snapshot_path = dates[0] # Get the newest folder

    args.snapshot_path = "dicewars/ai/xfrejl00/snapshots/" + args.snapshot_path + "/"

    if os.path.exists(args.snapshot_path) == False or os.path.isfile(args.snapshot_path + "saved_weights.h5") == False:
        print("ERROR: Can't find file with saved weights.")
        sys.exit(1)

    # Load the weights and start training the model
    model.train(weights_path=args.snapshot_path)
else: # Training from the start
    model.train()