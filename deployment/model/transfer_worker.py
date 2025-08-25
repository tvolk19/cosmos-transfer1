from imaginaire.utils.io import save_image_or_video
from cosmos_transfer2.configs.config_transfer_control2world import TRANSFER2_CONTROL2WORLD_PIPELINE_2B
from cosmos_transfer2.pipelines.transfer_control2world_pipeline import TransferControl2WorldPipeline
from imaginaire.constants import (
    COSMOS_TRANSFER2_CHECKPOINT_PATH_DEPTH,
    COSMOS_TRANSFER2_CHECKPOINT_PATH_VIZ,
    COSMOS_TRANSFER2_CHECKPOINT_PATH_SEG,
    COSMOS_TRANSFER2_CHECKPOINT_PATH_EDGE,
    COSMOS_TRANSFER2_CHECKPOINT_PATH_MV,
    T5_MODEL_DIR,
)
from deployment.model.model_config import Config
from cosmos_gradio.model_ipc.deployment_env import DeploymentEnv

import os
import json
from imaginaire.utils import log
from cosmos_transfer2.pipelines.utils import (
    get_prompt_from_path,
)
import gc
import torch

neg_prompt = "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all."

checkpoint_dict = {
    "depth": COSMOS_TRANSFER2_CHECKPOINT_PATH_DEPTH,
    "vis": COSMOS_TRANSFER2_CHECKPOINT_PATH_VIZ,
    "seg": COSMOS_TRANSFER2_CHECKPOINT_PATH_SEG,
    "edge": COSMOS_TRANSFER2_CHECKPOINT_PATH_EDGE,
    "mv": COSMOS_TRANSFER2_CHECKPOINT_PATH_MV,
}


class TransferValidator:
    def extract_params(self, json_data):
        hint_key = []
        args = {}
        control_modalities = {
            "edge": None,
            "vis": None,
            "depth": None,
            "seg": None,
        }

        for key, value in json_data.items():
            if key in ["edge", "depth", "seg", "vis", "hdmap_bbox"]:
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Invalid controlnet_specs value for '{key}': Expected dictionary, got {type(value)}"
                    )

                hint_key.append(key)
                for sub_key, sub_value in value.items():
                    if sub_key == "control_path":
                        control_modalities[key] = sub_value
                    # elif sub_key == "control_folder":
                    #     args[f"input_control_folder_{key}"] = sub_value
                    elif key == "edge" and sub_key == "preset_edge_threshold":
                        args["preset_edge_threshold"] = sub_value
                    else:
                        raise ValueError(f"Unknown controlnet_specs arg {sub_key} for hint key {key}")
            elif key == "prompt_path":
                args["prompt"] = get_prompt_from_path(value)
            else:
                args[key] = value

        args["hint_key"] = hint_key
        args["input_control_video_paths"] = control_modalities
        return args

    def validate_params(
        self,
        prompt: str,
        video_path: str,
        guidance: int = 7,
        seed: int = 1,
        resolution: str = "720",
        use_neg_prompt: bool = True,
        control_weight: float = 1.0,
        sigma_max: float | None = None,
        hint_key: list[str] = ["edge"],
        input_control_video_paths: dict[str, str] = {},
        show_control_condition: bool = False,
        show_input: bool = False,
        image_context_path: str = None,
        keep_input_resolution: bool = True,
        preset_edge_threshold: str = "medium",
        preset_blur_strength: str = "medium",
        num_video_frames_per_chunk: int = 93,
        num_conditional_frames: int = 1,
        output_dir: str = "outputs/",
    ):
        """inference function stub for testing validity of parameters"""
        valid_edge_thresholds = ["none", "very_low", "low", "medium", "high", "very_high"]

        pass

    def parse_and_validate(self, controlnet_specs):
        args_dict = self.extract_params(controlnet_specs)
        full_dict = self.validate_params(
            **args_dict,
        )
        full_dict = args_dict

        log.info(json.dumps(full_dict, indent=4))

        return full_dict


class TransferWorker:
    def __init__(self, num_gpus, checkpoint_dir, model_name="depth"):
        self.device_rank = 0
        self.process_group = 0
        self.num_gpus = num_gpus
        if num_gpus > 1:
            from megatron.core import parallel_state
            from imaginaire.utils import distributed

            log.info(f"Initializing distributed environment with {num_gpus} GPUs for context parallelism")

            # Check if distributed environment is already initialized
            assert not parallel_state.is_initialized()
            distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=num_gpus)
            log.info(f"Context parallel group initialized with {num_gpus} GPUs")
            self.process_group = parallel_state.get_context_parallel_group()
            self.device_rank = distributed.get_rank(self.process_group)

        # Create the video generation pipeline.
        self.pipe = TransferControl2WorldPipeline.from_config(
            config=TRANSFER2_CONTROL2WORLD_PIPELINE_2B,
            dit_path=checkpoint_dict[model_name],
            text_encoder_path=T5_MODEL_DIR,
            negative_prompt=neg_prompt,
        )
        self.video_save_name = "output"

    def infer(self, args: dict):
        output_dir = args.get("output_dir", "/mnt/pvc/gradio_output")
        args.pop("output_dir", None)  # Remove output_dir from args

        video, fps = self.pipe.generate_control2world(**args)

        if self.device_rank == 0:
            prompt = args.get("prompt", "")
            prompt_save_path = os.path.join(output_dir, f"{self.video_save_name}.txt")
            video_save_path = os.path.join(output_dir, f"{self.video_save_name}.mp4")

            with open(prompt_save_path, "wb") as f:
                f.write(prompt.encode("utf-8"))

            log.info(f"Saved prompt to {prompt_save_path}")
            save_image_or_video(video, video_save_path, fps=fps)
            log.info(f"Saved video to {video_save_path}")

    def __del__(self):
        self.pipe = None
        if self.num_gpus > 1:
            from megatron.core import parallel_state

            parallel_state.destroy_model_parallel()
            import torch.distributed as dist

            dist.destroy_process_group()


def create_worker(create_model=True):
    """Factory function to create transfer pipeline and validator.

    Args:
        cfg: Configuration object with model settings including checkpoint_dir
        create_model (bool): Whether to actually create the model pipeline (default: True)

    Returns:
        tuple: (pipeline, validator) - TransferPipeline instance and TransferValidator
    """
    cfg = DeploymentEnv()
    pipeline = None
    if create_model:
        pipeline = TransferWorker(
            num_gpus=cfg.num_gpus,
            checkpoint_dir=cfg.checkpoint_dir,
            model_name=cfg.model_name,
        )
        gc.collect()
        torch.cuda.empty_cache()

    validator = TransferValidator()
    return pipeline, validator
