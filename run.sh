#!/bin/bash

#export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR=/mnt/pvc/cosmos-transfer1

export NUM_GPU="${NUM_GPU:=1}"

PYTHONPATH=$(pwd) python3 cosmos_transfer1/diffusion/inference/transfer_pipeline.py
# PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer_pipeline.py

#PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py

# export NUM_GPU="${NUM_GPU:=8}"
# PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
#     --checkpoint_dir $CHECKPOINT_DIR \
#     --video_save_folder outputs/example1_single_control_edge \
#     --controlnet_specs assets/inference_cosmos_transfer1_single_control_edge.json \
#     --offload_text_encoder_model \
#     --offload_guardrail_models \
#     --num_gpus $NUM_GPU
