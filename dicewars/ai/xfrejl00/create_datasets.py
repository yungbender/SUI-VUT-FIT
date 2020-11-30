import argparse
import random
import os

VAL_DATASET_RATE = 0.1 # Share of validation samples compared to training samples

parser = argparse.ArgumentParser()
parser.add_argument("--snapshot_path", dest="dest_folder", required=True, action="store", help="Choose from which folder the raw dataset data will be loaded.")
parser.add_argument("--dqn", dest="dqn", action="store_true", help="Whether to take DQN data and labels or classifier data and labels.")
parser.add_argument("--clear", dest="clear", action="store_true", help="Whether to clear dataset files prior to appending new data.")
args = parser.parse_args()

dataset_path = "dicewars/ai/xfrejl00/" 
classifier_path = dataset_path + "snapshots/" + args.dest_folder + "/classifier/"
dqn_path = dataset_path + "snapshots/" + args.dest_folder + "/dqn/"

if args.dqn:
    if not os.path.exists(dqn_path):
        print("Selected snapshot folder doesn't contain folder for DQN dataset!")
        exit(1)

    if not os.path.isfile(dqn_path + "positives.txt") or not os.path.isfile(dqn_path + "negatives.txt"):
        print("Selected snapshot folder doesn't have positives.txt or negatives.txt file!")
        exit(1) 
else:
    if not os.path.exists(classifier_path):
        print("Selected snapshot folder doesn't contain folder for classifier dataset!")
        exit(1)

    if not os.path.isfile(classifier_path + "positives.txt") or not os.path.isfile(classifier_path + "negatives.txt"):
        print("Selected snapshot folder doesn't have positives.txt or negatives.txt file!")
        exit(1)

if args.clear: # Clear dataset files first
    if args.dqn:
        with open(dataset_path + "positives_dqn.trn", "w"), open(dataset_path + "positives_dqn.val", "w"), open(dataset_path + "negatives_dqn.trn", "w"), open(dataset_path + "negatives_dqn.val", "w"):
            pass
    else:
        with open(dataset_path + "positives.trn", "w"), open(dataset_path + "positives.val", "w"), open(dataset_path + "negatives.trn", "w"), open(dataset_path + "negatives.val", "w"):
            pass

if args.dqn:
    # Create positives
    with open(dqn_path + "positives.txt", "r") as raw_positives, open(dataset_path + "positives_dqn.trn", "a+") as trn_positives, open(dataset_path + "positives_dqn.val", "a+") as val_positives:
        for line in raw_positives:
            if random.uniform(0, 1) > VAL_DATASET_RATE: # Line goes to train dataset
                trn_positives.write(line)
            else: # Line goes to val dataset
                val_positives.write(line)
        
    # Create negatives
    with open(dqn_path + "negatives.txt", "r") as raw_negatives, open(dataset_path + "negatives_dqn.trn", "a+") as trn_negatives, open(dataset_path + "negatives_dqn.val", "a+") as val_negatives:
        for line in raw_negatives:
            if random.uniform(0, 1) > VAL_DATASET_RATE: # Line goes to train dataset
                trn_negatives.write(line)
            else: # Line goes to val dataset
                val_negatives.write(line)
       
else:
    # Create positives
    with open(classifier_path + "positives.txt", "r") as raw_positives, open(dataset_path + "positives.trn", "a+") as trn_positives, open(dataset_path + "positives.val", "a+") as val_positives:
        for line in raw_positives:
            if random.uniform(0, 1) > VAL_DATASET_RATE: # Line goes to train dataset
                trn_positives.write(line)
            else: # Line goes to val dataset
                val_positives.write(line)
        
    # Create negatives
    with open(classifier_path + "negatives.txt", "r") as raw_negatives, open(dataset_path + "negatives.trn", "a+") as trn_negatives, open(dataset_path + "negatives.val", "a+") as val_negatives:
        for line in raw_negatives:
            if random.uniform(0, 1) > VAL_DATASET_RATE: # Line goes to train dataset
                trn_negatives.write(line)
            else: # Line goes to val dataset
                val_negatives.write(line)