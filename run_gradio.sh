#!/bin/bash

#export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR=/mnt/pvc/cosmos-transfer1
export NUM_GPU="${NUM_GPU:=1}"
PYTHONPATH=$(pwd) python gradio_app.py
