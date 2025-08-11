#!/bin/bash

#export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR=/mnt/pvc/checkp_seg/cosmos-transfer1
export NUM_GPU="${NUM_GPU:=8}"
PYTHONPATH=$(pwd) python server/test_model_server.py
