#!/usr/bin/env bash

python3 -m venv ../env-sui
source ../env-sui/bin/activate
pip install -r requirements.txt
mkdir dicewars/logs # log file creation doesn't check for folder existence for some reason
