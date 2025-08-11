#!/bin/bash

#export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR=/mnt/pvc/checkp_seg/cosmos-transfer1
export NUM_GPU="${NUM_GPU:=4}"
# PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
#     --checkpoint_dir $CHECKPOINT_DIR \
#     --video_save_folder outputs/example1_single_control_edge \
#     --controlnet_specs assets/inference_cosmos_transfer1_single_control_edge.json \
#     --offload_text_encoder_model \
#     --offload_guardrail_models \
#     --num_gpus $NUM_GPU \
#     --num_steps -1

# vis
# PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
#     --checkpoint_dir $CHECKPOINT_DIR \
#     --video_save_folder outputs/example1_single_control_vis_8gpu \
#     --controlnet_specs assets/inference_cosmos_transfer1_single_control_vis.json \
#     --offload_text_encoder_model \
#     --offload_guardrail_models \
#     --num_gpus $NUM_GPU

PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/inference_cosmos_transfer1_uniform_weights_4gpu \
    --controlnet_specs assets/inference_cosmos_transfer1_uniform_weights.json \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus $NUM_GPU

# export PROMPT="The video is captured from a camera mounted on a car. The camera is facing forward. The video showcases a scenic golden-hour drive through a suburban area, bathed in the warm, golden hues of the setting sun. The dashboard camera captures the play of light and shadow as the sun’s rays filter through the trees, casting elongated patterns onto the road. The streetlights remain off, as the golden glow of the late afternoon sun provides ample illumination. The two-lane road appears to shimmer under the soft light, while the concrete barrier on the left side of the road reflects subtle warm tones. The stone wall on the right, adorned with lush greenery, stands out vibrantly under the golden light, with the palm trees swaying gently in the evening breeze. Several parked vehicles, including white sedans and vans, are seen on the left side of the road, their surfaces reflecting the amber hues of the sunset. The trees, now highlighted in a golden halo, cast intricate shadows onto the pavement. Further ahead, houses with red-tiled roofs glow warmly in the fading light, standing out against the sky, which transitions from deep orange to soft pastel blue. As the vehicle continues, a white sedan is seen driving in the same lane, while a black sedan and a white van move further ahead. The road markings are crisp, and the entire setting radiates a peaceful, almost cinematic beauty. The golden light, combined with the quiet suburban landscape, creates an atmosphere of tranquility and warmth, making for a mesmerizing and soothing drive."
# PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
#     --checkpoint_dir $CHECKPOINT_DIR \
#     --video_save_name output_video \
#     --video_save_folder outputs/sample_av_multi_control \
#     --prompt "$PROMPT" \
#     --sigma_max 80 \
#     --offload_text_encoder_model --is_av_sample \
#     --controlnet_specs assets/sample_av_multi_control_spec.json \
#     --num_gpus $NUM_GPU


# {
#     "prompt": "The video is captured from a camera mounted on a car. The camera is facing forward. The video showcases a scenic golden-hour drive through a suburban area, bathed in the warm, golden hues of the setting sun. The dashboard camera captures the play of light and shadow as the sun’s rays filter through the trees, casting elongated patterns onto the road. The streetlights remain off, as the golden glow of the late afternoon sun provides ample illumination. The two-lane road appears to shimmer under the soft light, while the concrete barrier on the left side of the road reflects subtle warm tones. The stone wall on the right, adorned with lush greenery, stands out vibrantly under the golden light, with the palm trees swaying gently in the evening breeze. Several parked vehicles, including white sedans and vans, are seen on the left side of the road, their surfaces reflecting the amber hues of the sunset. The trees, now highlighted in a golden halo, cast intricate shadows onto the pavement. Further ahead, houses with red-tiled roofs glow warmly in the fading light, standing out against the sky, which transitions from deep orange to soft pastel blue. As the vehicle continues, a white sedan is seen driving in the same lane, while a black sedan and a white van move further ahead. The road markings are crisp, and the entire setting radiates a peaceful, almost cinematic beauty. The golden light, combined with the quiet suburban landscape, creates an atmosphere of tranquility and warmth, making for a mesmerizing and soothing drive.",
#     "sigma_max": 80,
#     "hdmap": {
#         "control_weight": 0.3,
#         "input_control": "assets/sample_av_multi_control_input_hdmap.mp4"
#     },
#     "lidar": {
#         "control_weight": 0.7,
#         "input_control": "assets/sample_av_multi_control_input_lidar.mp4"
#     }
# }

# "input_video_path": "/mnt/pvc/gradio/uploads/upload_20250801_163255/input_image.mp4",
# "prompt": "The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.",
# "negative_prompt": "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all.",
# "guidance": 7.0,
# "num_steps": 35,
# "seed": 1,
# "sigma_max": 70.0,
# "blur_strength": "medium",
# "canny_threshold": "medium",
# "edge": {
# "control_weight": 1.0
# }