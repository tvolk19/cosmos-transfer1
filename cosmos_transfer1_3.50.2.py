"""
Gradio app that invokes cosmos-transfer1/cosmos_transfer1/diffusion/inference/transfer.py directly.

To use this app, it must be copied to the cosmos-transfer1 repository and run from there.

This app does less UI request construction than the 5.33.2 app: only requiring a video and JSON string, which is then parsed
into a controlnet spec and passed to transfer.demo. However, this is still pretty tightly coupled to the cosmos-transfer1
repository.

Tested with gradio==3.50.2
"""

import datetime
import json
import os
import random
import shutil
import string
import traceback
import zoneinfo
from typing import Optional

import cosmos_transfer1.diffusion.inference.transfer as transfer
from cosmos_transfer1.diffusion.inference import inference_utils
from cosmos_transfer1.utils import log

import gradio as gr


def create_gradio_blocks(checkpoint_dir, output_dir):

    def _infer(input_video, input_text: str) -> tuple[Optional[str], Optional[str]]:
        """
        Inference function which is called by the gradio app.

        - takes the inputs from the UI (video and text)
        - re-shapes them to be compatible with transfer.demo
        - runs transfer.demo
        - returns the output video path and/or status message

        Args:
            input_video: Path to the input video.
            input_text: JSON string containing the input parameters.

        Returns:
            Tuple containing the path to the output video and the status message.
        """
        try:
            if not input_video:
                log.error("No input video provided")
                raise ValueError("No input video provided. Please upload a video.")

            if not input_text:
                log.error("No input text provided")
                raise ValueError("No input text provided. Please provide a valid JSON string.")

            # Parse input text as JSON
            input_json = json.loads(input_text)

            # Create unique output folder
            timestamp = datetime.datetime.now(zoneinfo.ZoneInfo("US/Pacific")).strftime("%Y-%m-%d_%H-%M-%S")
            random_generation_id = "".join(random.choices(string.ascii_lowercase, k=4))
            output_folder = os.path.join(output_dir, f"generation_{timestamp}_{random_generation_id}")
            os.makedirs(output_folder, exist_ok=True)

            # set gradio values (either static values or ones generated for gradio)
            input_json["checkpoint_dir"] = checkpoint_dir
            input_json["input_video_path"] = input_video
            input_json["video_save_folder"] = output_folder
            input_json["video_save_name"] = "output"

            # Write controlnet_specs to file (so we can store it in the CLI args)
            controlnet_specs_path = os.path.join(output_folder, "input_controlnet_specs.json")
            with open(controlnet_specs_path, "w") as f:
                json.dump(input_json, f, indent=2)

            # Create arguments namespace
            parser = transfer.create_argument_parser()
            args = parser.parse_args(["--controlnet_specs", controlnet_specs_path])
            controlnet_specs, other_json_args = inference_utils.load_controlnet_specs(args)
            args = transfer.apply_overrides(parser, args, other_json_args)

            # Copy input video to output folder for reference
            input_copy_path = os.path.join(output_folder, f"input_{os.path.basename(input_video)}")
            shutil.copy2(input_video, input_copy_path)

            # Run the existing demo function
            video_save_paths = transfer.demo(args, controlnet_specs)

            # Check if output was generated
            if len(video_save_paths) == 0:
                return None, f"Generation failed - no output video was created\nCheck folder: {output_folder}"
            if len(video_save_paths) > 1:
                log.warning(f"Multiple output videos were created. Only the first one will be used: {video_save_paths}")

            # Build resonse from first output video
            output_video_path = video_save_paths[0]
            output_prompt_exists = False
            final_prompt = ""
            prompt_paths = [os.path.join(output_folder, "prompt.txt"), os.path.join(output_folder, "output.txt")]
            for prompt_path in prompt_paths:
                if os.path.exists(prompt_path):
                    log.info(f"Found output prompt: {prompt_path}")
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        final_prompt = f.read().strip()
                        output_prompt_exists = True
                        break
            if not output_prompt_exists:
                log.warning(f"No prompt was generated. Using the prompt from the input JSON: {input_json.get('prompt', '')}")
                final_prompt = input_json.get("prompt", transfer.DEFAULT_PROMPT)

            return output_video_path, f"Video generated successfully!\nOutput saved to: {output_folder}\nFinal prompt: {final_prompt}"

        except Exception as e:
            log.error(f"Error during generation:\n{traceback.format_exc()}")
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
                input_text = gr.Textbox(
                    label="Controlnet Specs JSON",
                    lines=10,
                    interactive=True,
                    placeholder=json.dumps(
                        {
                            "prompt": transfer.DEFAULT_PROMPT[:20] + "...",
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

        generate_btn.click(fn=_infer, inputs=[input_video, input_text], outputs=[output_video, output_text])

    concurrency_limit = int(os.environ.get("GRADIO_CONCURRENCY_LIMIT", 1))
    max_queue_size = int(os.environ.get("GRADIO_MAX_QUEUE_SIZE", 20))
    try:
        status_update_rate = float(os.environ.get("GRADIO_STATUS_UPDATE_RATE", 1.0))
    except ValueError:
        status_update_rate = "auto"
    log.info(f"Configuring queue with {concurrency_limit=} {max_queue_size=} {status_update_rate=}")

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

    log.init_loguru_stdout()
    log.info(f"Starting Gradio Cosmos-Transfer1 App - {server_name=} {server_port=} {allowed_paths=} {save_dir=}")

    blocks = create_gradio_blocks(checkpoint_dir="checkpoints", output_dir=save_dir)
    blocks.launch(
        allowed_paths=allowed_paths,
        server_name=server_name,
        server_port=server_port,
        share=False,
    )
