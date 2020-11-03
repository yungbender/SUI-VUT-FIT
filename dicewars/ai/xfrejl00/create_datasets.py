import argparse
import random
import os

VAL_DATASET_RATE = 0.1 # Share of validation samples compared to training samples

parser = argparse.ArgumentParser()
parser.add_argument("--snapshot_path", dest="dest_folder", required=True, action="store", help="Choose from which folder the raw dataset data will be loaded.")
parser.add_argument("--clear", dest="clear", action="store_true", help="Whether to clear dataset files prior to appending new data.")
args = parser.parse_args()

dataset_path = "dicewars/ai/xfrejl00/" 
snapshot_path = dataset_path + "snapshots/" + args.dest_folder + "/"
if not os.path.exists(snapshot_path):
    print("Selected snapshot folder doesn't exist!")
    exit(1)

if not os.path.isfile(snapshot_path + "positives.txt") or not os.path.isfile(snapshot_path + "negatives.txt"):
    print("Selected snapshot folder doesn't have positives.txt or negatives.txt file!")
    exit(1)

if args.clear: # Clear dataset files first
    with open(dataset_path + "positives.trn", "w"), open(dataset_path + "positives.val", "w"), open(dataset_path + "negatives.trn", "w"), open(dataset_path + "negatives.val", "w"):
        pass

# Create positives
with open(snapshot_path + "positives.txt", "r") as raw_positives, open(dataset_path + "positives.trn", "a+") as trn_positives, open(dataset_path + "positives.val", "a+") as val_positives:
    for line in raw_positives:
        if random.uniform(0, 1) > VAL_DATASET_RATE: # Line goes to train dataset
            trn_positives.write(line)
        else: # Line goes to val dataset
            val_positives.write(line)
    
# Create negatives
with open(snapshot_path + "negatives.txt", "r") as raw_negatives, open(dataset_path + "negatives.trn", "a+") as trn_negatives, open(dataset_path + "negatives.val", "a+") as val_negatives:
    for line in raw_negatives:
        if random.uniform(0, 1) > VAL_DATASET_RATE: # Line goes to train dataset
            trn_negatives.write(line)
        else: # Line goes to val dataset
            val_negatives.write(line)