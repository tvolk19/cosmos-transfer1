"""
Gradio app that invokes cosmos-transfer1/cosmos_transfer1/diffusion/inference/transfer.py via subprocess.

This app was an experiment to see if we could invoke torchrun via subprocess. This experiment worked; we
were able to spawn multi-gpu inference jobs via subprocess. However, there were 2 issues:
1. Spawning a torchrun subprocess for cosmos-transfer1 incurs overhead to load the model on every inference request.
2. Inference requests seemed to timeout after several minutes.

Tested with gradio==3.50.2
"""

import datetime
import json
import os
import random
import shlex
import shutil
import string
import subprocess
import traceback
import zoneinfo
from typing import Optional

from loguru import logger

import gradio as gr


def create_gradio_blocks(path_to_inference_script: str, checkpoint_dir: str, output_dir: str) -> gr.Blocks:
    """
    Creates the gradio blocks for the app.

    Args:
        path_to_inference_script: Path to the inference script (e.g. /mnt/pvc/gradio/cosmos-transfer1/cosmos_transfer1/diffusion/inference/transfer.py)
        checkpoint_dir: Path to the checkpoint directory (e.g. /mnt/pvc/gradio/cosmos-transfer1/checkpoints)
        output_dir: Path to the output directory (e.g. /mnt/pvc/gradio/cosmos-transfer1/outputs)

    Returns:
        Gradio blocks object.
    """

    def _infer(input_video, input_text_cli_arguments: str, input_text_controlnet_specs_json: str) -> tuple[Optional[str], Optional[str]]:
        """
        Inference function which is called by the gradio app.

        - takes the inputs from the UI (video and text)
        - applies some overrides to the input arguments and controlnet specs
        - saves the input video and controlnet specs to the output folder
        - runs the inference script in a subprocess
        - returns the output video path and/or status message

        Args:
            input_video: Path to the input video.
            input_text_cli_arguments: Command line arguments.
            input_text_controlnet_specs_json: Controlnet specs JSON.

        Returns:
            Tuple containing the path to the output video and the status message.
        """
        try:

            def _read_input(input_video: str, input_text_controlnet_specs_json: str) -> tuple[str, dict]:
                if not input_video:
                    logger.error("No input_video provided")
                    raise ValueError("No input video provided. Please upload a video.")

                if not input_text_controlnet_specs_json:
                    logger.error("No input_text_controlnet_specs_json provided")
                    raise ValueError("No input text controlnet specs provided. Please provide a valid JSON string.")

                input_json = json.loads(input_text_controlnet_specs_json)

                logger.info(f"{input_video=}")
                logger.info(f"{input_json=}")
                logger.info("Input validated")
                return input_video, input_json

            def _create_output_folder(output_dir: str) -> str:
                timestamp = datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).strftime("%Y-%m-%d_%H-%M-%S")
                random_generation_id = "".join(random.choices(string.ascii_lowercase, k=4))
                output_folder = os.path.join(output_dir, f"generation_{timestamp}_{random_generation_id}")
                os.makedirs(output_folder, exist_ok=True)
                logger.info(f"Output folder created: {output_folder}")
                return output_folder

            def _override_args_and_controlnet(cli_args: str, controlnet_json: dict, output_folder: str) -> tuple[list[str], dict]:
                controlnet_specs_path = _controlnet_specs_path(output_folder)
                overrides = {
                    "checkpoint_dir": checkpoint_dir,
                    "input_video_path": input_video,
                    "video_save_folder": output_folder,
                    "video_save_name": "output",
                    "controlnet_specs": controlnet_specs_path,
                }

                controlnet_json.update(overrides)

                cli_args_list = shlex.split(cli_args)
                final_cli_args = []
                i = 0
                while i < len(cli_args_list):
                    arg = cli_args_list[i]
                    key = arg.lstrip("-")
                    logger.info(f"{arg=} {key=} {final_cli_args=}")

                    # arg is not a flag (i.e. ['value']) => add it to the final_cli_args and continue
                    if not arg.startswith("-"):
                        final_cli_args.append(arg)
                        i += 1
                        continue

                    # arg is a flag => it's either: ['--key1=value'] or ['--key1', 'value']

                    # arg = '--key=value' => replace the value if its key is in overrides
                    if "=" in arg:
                        equals_index = arg.find("=")
                        key = arg[0:equals_index].lstrip("-")
                        if key in overrides:
                            final_cli_args.append(arg[0:equals_index] + "=" + overrides[key])
                        else:
                            final_cli_args.append(arg)
                        i += 1
                        continue

                    # arg = '--key' => skip next arg if key is in overrides
                    key = arg.lstrip("-")
                    if i + 1 < len(cli_args_list) and key in overrides:
                        final_cli_args.append(arg)
                        final_cli_args.append(overrides[key])
                        i += 2
                        continue

                    # arg = '--key' => without an override => add it and continue
                    final_cli_args.append(arg)
                    i += 1
                    continue

                logger.info(f"CLI args: {final_cli_args}")
                logger.info(f"Controlnet json: {controlnet_json}")
                logger.info("Args and controlnet specs overridden")

                return final_cli_args, controlnet_json

            def _controlnet_specs_path(output_folder: str) -> str:
                return os.path.join(output_folder, "input_controlnet_specs.json")

            def _save_controlnet_specs(input_json: dict, output_folder: str) -> str:
                path = _controlnet_specs_path(output_folder)
                with open(path, "w") as f:
                    json.dump(input_json, f, indent=2)

                logger.info(f"Controlnet specs saved to: {path}")

                return path

            def _save_input_video(input_video: str, output_folder: str) -> str:
                path = os.path.join(output_folder, f"input_{os.path.basename(input_video)}")
                shutil.copy2(input_video, path)

                logger.info(f"Input video saved to: {path}")

                return path

            def _num_gpu() -> int:

                def _num_gpu_from_env() -> Optional[int]:
                    num_gpu_from_env = None
                    try:
                        num_gpu_from_env = int(os.environ.get("NUM_GPU"))  # type: ignore
                    except Exception:
                        logger.warning("os.environ.NUM_GPU invalid")
                    logger.info(f"num_gpu_from_env: {num_gpu_from_env=}")
                    return num_gpu_from_env

                def _num_gpu_from_cli_args(cli_args: list[str]) -> Optional[int]:
                    num_gpu_from_cli_args = None
                    for i, arg in enumerate(cli_args):
                        logger.info(f"{i=} {arg=}")
                        try:
                            if arg == "--num_gpus":
                                num_gpu_from_cli_args = int(cli_args[i + 1])
                                break
                            if arg.startswith("--num_gpus="):
                                num_gpu_from_cli_args = int(arg.split("=")[1])
                                break
                        except Exception:
                            logger.warning(f"cli_args['--num_gpus'] invalid or unset: {cli_args[i:i + 1]}")
                    logger.info(f"num_gpu_from_cli_args: {num_gpu_from_cli_args=}")
                    return num_gpu_from_cli_args

                num_gpu = _num_gpu_from_cli_args(cli_args) or _num_gpu_from_env() or 1
                logger.info(f"Subprocess will use NUM_GPU={num_gpu}")
                return num_gpu

            def _run_inference_subprocess(cli_args: list[str]):
                num_gpu = _num_gpu()

                # Build environment for subprocess
                env = os.environ.copy()
                env["PYTHONPATH"] = os.getcwd()
                env["NUM_GPU"] = str(num_gpu)
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(0, num_gpu)))

                cmd = ["torchrun", f"--nproc_per_node={num_gpu}", "--nnodes=1", "--node_rank=0", path_to_inference_script, *cli_args]

                logger.info("Running inference subprocess...")
                logger.info(f"cmd={'\n'.join(cmd)}")
                logger.info(f"env['PYTHONPATH']={env['PYTHONPATH']}")
                logger.info(f"env['NUM_GPU']={env['NUM_GPU']}")
                logger.info(f"env['CUDA_VISIBLE_DEVICES']={env['CUDA_VISIBLE_DEVICES']}")

                result = subprocess.run(cmd, capture_output=False, check=True, env=env)

                logger.info(f"Inference subprocess result: {result}")

                return result

            def _get_output_video_path(output_folder: str) -> str:
                expected_path = os.path.join(output_folder, "output.mp4")
                if os.path.exists(expected_path):
                    logger.info(f"Found output video: {expected_path}")
                    return expected_path
                logger.warning(f"No output video found at {expected_path}. Returning empty string.")
                return ""

            def _get_output_prompt(output_folder: str) -> str:
                possible_prompt_paths = [os.path.join(output_folder, "prompt.txt"), os.path.join(output_folder, "output.txt")]
                for prompt_path in possible_prompt_paths:
                    if os.path.exists(prompt_path):
                        logger.info(f"Found output prompt: {prompt_path}")
                        with open(prompt_path, "r", encoding="utf-8") as f:
                            return f.read().strip()
                logger.warning(f"No output prompt found in {possible_prompt_paths}. Returning empty string.")
                return ""

            input_video, controlnet_json = _read_input(input_video, input_text_controlnet_specs_json)
            output_folder = _create_output_folder(output_dir)
            cli_args, controlnet_json = _override_args_and_controlnet(input_text_cli_arguments, controlnet_json, output_folder)
            _save_controlnet_specs(controlnet_json, output_folder)
            _save_input_video(input_video, output_folder)
            _run_inference_subprocess(cli_args)
            output_video_path = _get_output_video_path(output_folder)
            output_prompt = _get_output_prompt(output_folder)

            if not output_video_path:
                return None, f"Generation failed - no output video was created\nCheck folder: {output_folder}"

            return output_video_path, f"Video generated successfully!\nOutput saved to: {output_folder}\nFinal prompt: {output_prompt}"

        except Exception:
            logger.error(f"Error during generation:\n{traceback.format_exc()}")
            return None, f"Error during generation:\n\n{traceback.format_exc()}"

    # Define the gradio interface (UI/API)
    with gr.Blocks(title="Cosmos-Transfer1 Video Generation", theme=gr.themes.Soft()) as blocks:
        gr.Markdown("# Cosmos-Transfer1: World Generation with Adaptive Multimodal Control")
        gr.Markdown("Upload a video and configure controls to generate a new video with the Cosmos-Transfer1 model.")
        gr.Markdown(f"**Output Directory**: {output_dir}")

        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.Video(
                    label="Input Video",
                    height=500,
                    interactive=True,
                )
                input_text_cli_arguments = gr.Textbox(
                    label="Command Line Arguments",
                    lines=10,
                    interactive=True,
                    placeholder="--offload_guardrail_models --upsample_prompt --seed 1 --guidance=5",
                )
                input_text_controlnet_specs_json = gr.Textbox(
                    label="Controlnet Specs JSON",
                    lines=10,
                    interactive=True,
                    placeholder=json.dumps(
                        {
                            "prompt": "A beautiful woman in a red dress...",
                            "edge": {"control_weight": 1.0},
                            "fps": 24,
                            "guidance": 5,
                            "num_steps": 35,
                            "offload_diffusion_transformer": True,
                            "offload_guardrail_models": True,
                            "offload_text_encoder_model": True,
                            "seed": 1,
                            "upsample_prompt": True,
                        },
                        indent=0,
                    ),
                )
                generate_btn = gr.Button("Run Inference", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_video = gr.Video(label="Generated Video", height=500)
                output_text = gr.Textbox(label="Output", lines=10)

        generate_btn.click(
            fn=_infer, inputs=[input_video, input_text_cli_arguments, input_text_controlnet_specs_json], outputs=[output_video, output_text]
        )

    concurrency_limit = int(os.environ.get("GRADIO_CONCURRENCY_LIMIT", 1))
    max_queue_size = int(os.environ.get("GRADIO_MAX_QUEUE_SIZE", 20))
    try:
        status_update_rate = float(os.environ.get("GRADIO_STATUS_UPDATE_RATE", 1.0))
    except ValueError:
        status_update_rate = "auto"
    logger.info(f"Configuring queue with {concurrency_limit=} {max_queue_size=} {status_update_rate=}")

    return blocks.queue(
        concurrency_count=concurrency_limit,
        max_size=max_queue_size,
        status_update_rate=status_update_rate,
    )


if __name__ == "__main__":
    allowed_paths = os.environ.get("GRADIO_ALLOWED_PATHS", "/mnt/pvc/gradio").split(",")
    save_dir = os.environ.get("GRADIO_SAVE_DIR", "/mnt/pvc/gradio/cosmos-transfer1")

    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", 8080))

    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints")
    path_to_inference_script = os.environ.get("PATH_TO_INFERENCE_SCRIPT", "cosmos_transfer1/diffusion/inference/transfer.py")

    logger.info(f"Starting Gradio Cosmos-Transfer1 App - {server_name=} {server_port=} {allowed_paths=} {save_dir=}")

    blocks = create_gradio_blocks(path_to_inference_script, checkpoint_dir, output_dir=save_dir)
    blocks.launch(
        allowed_paths=allowed_paths,
        server_name=server_name,
        server_port=server_port,
        share=False,
    )
