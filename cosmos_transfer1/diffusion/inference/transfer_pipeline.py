#!/usr/bin/env python3
"""
Refactored pipeline classes for Cosmos Transfer generation.
Contains base class and derived class for pipeline initialization and generation.
"""
import argparse
import copy
import json
import os

# No typing imports needed

from cosmos_transfer1.checkpoints import (
    VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    BASE_7B_CHECKPOINT_PATH,
)
from cosmos_transfer1.diffusion.inference.inference_utils import validate_controlnet_specs
from cosmos_transfer1.diffusion.inference.preprocessors import Preprocessors
from cosmos_transfer1.diffusion.inference.world_generation_pipeline import DiffusionControl2WorldGenerationPipeline
from cosmos_transfer1.utils import log
from cosmos_transfer1.utils.io import save_video

# from typing import Dict, Any

# valid_hint_keys = {"vis", "seg", "edge", "depth", "keypoint", "upscale", "hdmap", "lidar"}


# def load_controlnet_specs(cfg) -> Dict[str, Any]:
#     with open(cfg, "r") as f:
#         controlnet_specs_in = json.load(f)

#     controlnet_specs = {}

#     for hint_key, config in controlnet_specs_in.items():
#         if hint_key in valid_hint_keys:
#             controlnet_specs[hint_key] = config
#         else:
#             if type(config) == dict:
#                 raise ValueError(f"Invalid hint_key: {hint_key}. Must be one of {valid_hint_keys}")
#             else:
#                 log.warning(f"Ignoring unknown control key: {hint_key}. Must be one of {valid_hint_keys}")
#                 continue
#     return controlnet_specs


class TransferPipeline:
    def __init__(
        self,
        num_gpus: int = 1,
        checkpoint_dir: str = "/mnt/pvc/cosmos-transfer1",
        # TODO control_inputs: str = "assets/inference_cosmos_transfer1_single_control_edge.json",
    ):

        self.pipeline = None
        self.preprocessors = None
        self.device_rank = 0
        self.process_group = None

        self.preprocessors = Preprocessors()

        if num_gpus > 1:
            from megatron.core import parallel_state
            from cosmos_transfer1.utils import distributed

            distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=num_gpus)
            self.process_group = parallel_state.get_context_parallel_group()
            self.device_rank = distributed.get_rank(self.process_group)

        self.control_inputs = self.create_controlnet_spec(checkpoint_dir=checkpoint_dir)
        # self.control_inputs = load_controlnet_specs(control_inputs)
        # self.control_inputs = validate_controlnet_specs(cfg, control_inputs)

        self.pipeline = DiffusionControl2WorldGenerationPipeline(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=BASE_7B_CHECKPOINT_PATH,
            control_inputs=self.control_inputs,
            process_group=self.process_group,
        )

    def create_controlnet_spec(
        self,
        checkpoint_dir: str = "/mnt/pvc/cosmos-transfer1",
        vis_weight=0,
        edge_weight=1,
        depth_weight=0,
        seg_weight=0,
        keypoint_weight=0,
    ):
        control_inputs = {}

        if vis_weight > 0:
            control_inputs["vis"] = {
                "ckpt_path": VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
                "control_weight": vis_weight,
            }

        if edge_weight > 0:
            control_inputs["edge"] = {
                "ckpt_path": os.path.join(checkpoint_dir, EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH),
                "control_weight": edge_weight,
            }

        if depth_weight > 0:
            control_inputs["depth"] = {"control_weight": depth_weight}

        if seg_weight > 0:
            control_inputs["seg"] = {"control_weight": seg_weight}

        if keypoint_weight > 0:
            control_inputs["keypoint"] = {"control_weight": keypoint_weight}

        log.info(f"control_inputs: {json.dumps(control_inputs, indent=4)}")

        return control_inputs

    def infer(self, cfg):

        # Run preprocessor
        log.info("Running preprocessor")
        current_control_inputs = copy.deepcopy(self.control_inputs)

        self.preprocessors(
            cfg.input_video_path,
            cfg.prompt,
            current_control_inputs,
            cfg.video_save_folder,
        )
        log.info(f"current_control_inputs: {json.dumps(current_control_inputs, indent=4)}")

        if hasattr(self.pipeline, "regional_prompts"):
            self.pipeline.regional_prompts = []
        if hasattr(self.pipeline, "region_definitions"):
            self.pipeline.region_definitions = []

        self.pipeline.offload_network = cfg.offload_diffusion_transformer
        self.pipeline.offload_text_encoder_model = cfg.offload_text_encoder_model
        self.pipeline.offload_guardrail_models = cfg.offload_guardrail_models
        self.pipeline.guidance = cfg.guidance
        self.pipeline.num_steps = cfg.num_steps
        self.pipeline.fps = cfg.fps
        self.pipeline.seed = cfg.seed
        self.pipeline.num_input_frames = cfg.num_input_frames
        self.pipeline.sigma_max = cfg.sigma_max
        self.pipeline.blur_strength = cfg.blur_strength
        self.pipeline.canny_threshold = cfg.canny_threshold
        self.pipeline.upsample_prompt = cfg.upsample_prompt
        self.pipeline.offload_prompt_upsampler = cfg.offload_prompt_upsampler

        batch_outputs = self.pipeline.generate(
            prompt=[cfg.prompt],
            video_path=[cfg.input_video_path],
            negative_prompt=cfg.negative_prompt,
            control_inputs=[current_control_inputs],
            save_folder=cfg.video_save_folder,
            batch_size=1,
        )

        if self.device_rank == 0:
            videos, final_prompts = batch_outputs
            for i, (video, prompt) in enumerate(zip(videos, final_prompts)):

                video_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.mp4")
                prompt_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.txt")
                os.makedirs(os.path.dirname(video_save_path), exist_ok=True)

                save_video(
                    video=video,
                    fps=cfg.fps,
                    H=video.shape[1],
                    W=video.shape[2],
                    video_save_quality=5,
                    video_save_path=video_save_path,
                )

                # Save prompt to text file alongside video
                with open(prompt_save_path, "wb") as f:
                    f.write(prompt.encode("utf-8"))

                log.info(f"Saved video to {video_save_path}")
                log.info(f"Saved prompt to {prompt_save_path}")

    @staticmethod
    def create_model_params(
        input_video="assets/example1_input_video.mp4",
        prompt="The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.",
        negative_prompt="The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all.",
        guidance=5,
        num_steps=35,
        seed=1,
        sigma_max=70.0,
        blur_strength="medium",
        canny_threshold="medium",
        output_folder="outputs/",
    ):
        args = argparse.Namespace()

        # for hint_key, config in controlnet_specs.items():
        #     if hint_key not in valid_hint_keys:
        #         raise ValueError(f"Invalid hint_key: {hint_key}. Must be one of {valid_hint_keys}")

        #     if not input_video_path and sigma_max < 80:
        #         raise ValueError("Must have 'input_video' specified if sigma_max < 80")

        #     if not input_video_path and "input_control" not in config:
        #         raise ValueError(
        #             f"{hint_key} controlnet must have 'input_control' video specified if no 'input_video' specified."
        #         )

        # Video and prompt settings
        args.input_video_path = input_video
        args.prompt = prompt
        args.negative_prompt = negative_prompt
        args.video_save_folder = output_folder
        args.video_save_name = "output"

        # Generation parameters
        args.guidance = guidance
        args.num_steps = num_steps
        args.seed = seed
        args.sigma_max = sigma_max
        args.blur_strength = blur_strength
        args.canny_threshold = canny_threshold

        # Model settings
        args.is_av_sample = False
        args.tokenizer_dir = "Cosmos-Tokenize1-CV8x8x8-720p"
        args.num_input_frames = 1
        args.fps = 24
        args.batch_size = 1
        # args.num_gpus = 1

        # Memory optimization
        args.offload_diffusion_transformer = True
        args.offload_text_encoder_model = True
        args.offload_guardrail_models = True
        args.upsample_prompt = False
        args.offload_prompt_upsampler = True

        return args

    def cleanup(self, cfg):
        """Clean up resources"""
        if cfg.num_gpus > 1:
            from megatron.core import parallel_state
            import torch.distributed as dist

            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


if __name__ == "__main__":
    pipeline = TransferPipeline(num_gpus=8)
    model_params = TransferPipeline.create_model_params()
    pipeline.infer(model_params)

    log.info("Inference complete****************************************")
    pipeline.infer(model_params)
