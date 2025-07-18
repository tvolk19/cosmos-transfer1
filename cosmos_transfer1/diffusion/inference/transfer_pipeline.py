#!/usr/bin/env python3

import argparse
import copy
import json
import os


from cosmos_transfer1.checkpoints import (
    VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    BASE_7B_CHECKPOINT_PATH,
)
from cosmos_transfer1.diffusion.inference.preprocessors import Preprocessors
from cosmos_transfer1.diffusion.inference.world_generation_pipeline import DiffusionControl2WorldGenerationPipeline
from cosmos_transfer1.utils import log
from cosmos_transfer1.utils.io import save_video
from cosmos_transfer1.utils.misc import force_gc

"""
pipeline class similar to demo function.
we can't use the demo function as the function discards the underlying pipeline after each inference.

This class creates the pipeline during initialization and keeps it alive for multiple inferences.
The pipeline is configured for to NOT offload the models in favor of speed.

For now we assume a fixed configuration of the controlnets for the lifetime of the pipeline.
This makes the config.json file for the controlnets a init parameter of the pipeline.

TODO can we load all controlnets at once? otherwise need to configure control nets

TODO batch support

TODO regional prompt support

TODO av support

"""


class TransferPipeline:
    def __init__(
        self, num_gpus: int = 1, checkpoint_dir: str = "/mnt/pvc/cosmos-transfer1", output_dir: str = "outputs/"
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

        self.output_dir = output_dir
        self.video_save_name = "output"

        self.pipeline = DiffusionControl2WorldGenerationPipeline(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=BASE_7B_CHECKPOINT_PATH,
            control_inputs=self.control_inputs,
            process_group=self.process_group,
            offload_network=False,
            offload_text_encoder_model=True,
            offload_guardrail_models=False,
            offload_prompt_upsampler=False,
            upsample_prompt=False,
            fps=24,
            num_input_frames=24,
        )
        force_gc("After initializing pipeline")

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
                "ckpt_path": os.path.join(checkpoint_dir, VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH),
                "control_weight": vis_weight,
            }

        if edge_weight > 0:
            control_inputs["edge"] = {
                "ckpt_path": os.path.join(checkpoint_dir, EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH),
                "control_weight": edge_weight,
            }

        if depth_weight > 0:
            control_inputs["depth"] = {
                "ckpt_path": os.path.join(checkpoint_dir, DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH),
                "control_weight": depth_weight,
            }

        if seg_weight > 0:
            control_inputs["seg"] = {
                "ckpt_path": os.path.join(checkpoint_dir, SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH),
                "control_weight": seg_weight,
            }

        if keypoint_weight > 0:
            control_inputs["keypoint"] = {
                "ckpt_path": os.path.join(checkpoint_dir, KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH),
                "control_weight": keypoint_weight,
            }

        log.info(f"control_inputs: {json.dumps(control_inputs, indent=4)}")

        return control_inputs

    def set_control_weights(
        self,
        vis_weight=0,
        edge_weight=1,
        depth_weight=0,
        seg_weight=0,
        keypoint_weight=0,
    ):
        if vis_weight > 0:
            if "vis" not in self.control_inputs:
                log.warning("Visual control network is not configured is this network. Ignoring vis_weight.")
            else:
                self.control_inputs["vis"]["control_weight"] = vis_weight

        if edge_weight > 0:
            if "edge" not in self.control_inputs:
                log.warning("Edge control network is not configured is this network. Ignoring edge_weight.")
            else:
                self.control_inputs["edge"]["control_weight"] = edge_weight

        if depth_weight > 0:
            if "depth" not in self.control_inputs:
                log.warning("Depth control network is not configured is this network. Ignoring depth_weight.")
            else:
                self.control_inputs["depth"]["control_weight"] = depth_weight

        if seg_weight > 0:
            if "seg" not in self.control_inputs:
                log.warning("Segment control network is not configured is this network. Ignoring seg_weight.")
            else:
                self.control_inputs["seg"]["control_weight"] = seg_weight

        if keypoint_weight > 0:
            if "keypoint" not in self.control_inputs:
                log.warning("Keypoint control network is not configured is this network. Ignoring keypoint_weight.")
            else:
                self.control_inputs["keypoint"]["control_weight"] = keypoint_weight

        log.info(f"control_inputs: {json.dumps(self.control_inputs, indent=4)}")

    def infer(self, params):

        self.set_control_weights(
            vis_weight=params.vis_weight,
            edge_weight=params.edge_weight,
            depth_weight=params.depth_weight,
            seg_weight=params.seg_weight,
            keypoint_weight=params.keypoint_weight,
        )

        # original code is creating deepcopy. are values touched?
        # TODO add control weights as inference parameter
        current_control_inputs = copy.deepcopy(self.control_inputs)
        log.info(f"current_control_inputs: {json.dumps(current_control_inputs, indent=4)}")

        log.info("Running preprocessor")
        self.preprocessors(
            params.input_video,
            params.prompt,
            current_control_inputs,
            self.output_dir,
        )
        force_gc("After running preprocessor")

        # TODO: add support for regional prompts and region definitions
        if hasattr(self.pipeline, "regional_prompts"):
            self.pipeline.regional_prompts = []
        if hasattr(self.pipeline, "region_definitions"):
            self.pipeline.region_definitions = []

        # WAR these inference parameters are for unknown reasons not part of the generate function
        self.pipeline.guidance = params.guidance
        self.pipeline.num_steps = params.num_steps
        self.pipeline.seed = params.seed
        self.pipeline.sigma_max = params.sigma_max
        self.pipeline.blur_strength = params.blur_strength
        self.pipeline.canny_threshold = params.canny_threshold

        batch_outputs = self.pipeline.generate(
            prompt=[params.prompt],
            video_path=[params.input_video],
            negative_prompt=params.negative_prompt,
            control_inputs=[current_control_inputs],
            save_folder=self.output_dir,
            batch_size=1,
        )
        force_gc("After generating batch outputs")

        if self.device_rank == 0:
            videos, final_prompts = batch_outputs
            for i, (video, prompt) in enumerate(zip(videos, final_prompts)):

                video_save_path = os.path.join(self.output_dir, f"{self.video_save_name}.mp4")
                prompt_save_path = os.path.join(self.output_dir, f"{self.video_save_name}.txt")
                os.makedirs(os.path.dirname(video_save_path), exist_ok=True)

                save_video(
                    video=video,
                    fps=self.pipeline.fps,
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
    def validate_params(
        input_video="assets/example1_input_video.mp4",
        prompt="The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.",
        negative_prompt="The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all.",
        vis_weight=0.5,
        edge_weight=0.5,
        depth_weight=0.5,
        seg_weight=0.5,
        keypoint_weight=0.5,
        guidance=5,
        num_steps=35,
        seed=1,
        sigma_max=70.0,
        blur_strength="medium",
        canny_threshold="medium",
    ):
        args = argparse.Namespace()

        if sigma_max < 80 and not input_video:
            raise ValueError("Must have 'input_video' specified if sigma_max < 80")

        # TODO: edge, canny need input_video
        # TODO depth, seg, keypoint need either input_video OR input_control
        # for hint_key, config in controlnet_specs.items():

        #     if not input_video and "input_control" not in config:
        #         raise ValueError(
        #             f"{hint_key} controlnet must have 'input_control' video specified if no 'input_video' specified."
        #         )

        # Video and prompt settings
        args.input_video = input_video
        args.prompt = prompt
        args.negative_prompt = negative_prompt

        # Control weights
        args.vis_weight = vis_weight
        args.edge_weight = edge_weight
        args.depth_weight = depth_weight
        args.seg_weight = seg_weight
        args.keypoint_weight = keypoint_weight

        # Generation parameters
        args.guidance = guidance
        args.num_steps = num_steps
        args.seed = seed
        args.sigma_max = sigma_max
        args.blur_strength = blur_strength
        args.canny_threshold = canny_threshold

        args_dict = {key: value for key, value in vars(args).items()}
        log.info(f"Model parameters: {json.dumps(args_dict, indent=4)}")

        return args, args_dict

    def cleanup(self, cfg):
        """Clean up resources"""
        if cfg.num_gpus > 1:
            from megatron.core import parallel_state
            import torch.distributed as dist

            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


if __name__ == "__main__":
    pipeline = TransferPipeline(num_gpus=int(os.environ.get("NUM_GPU", 1)))
    model_params, _ = TransferPipeline.validate_params()
    pipeline.infer(model_params)

    log.info("Inference complete****************************************")
    pipeline.infer(model_params)
