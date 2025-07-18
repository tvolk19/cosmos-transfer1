#!/bin/bash

export CHECKPOINT_DIR=/mnt/pvc/cosmos-transfer1
export NUM_GPU="${NUM_GPU:=1}"
PYTHONPATH=$(pwd) python model_server.py
