"""
Gradio app that invokes cosmos-transfer1/cosmos_transfer1/diffusion/inference/transfer.py directly.

To use this app, it must be copied to the cosmos-transfer1 repository and run from there.

This app does more UI request construction than is ideal. It's included here for historical reference.

Tested with gradio==5.33.2
"""

import argparse
import os
import shutil
from datetime import datetime
from typing import Optional

import cosmos_transfer1.diffusion.inference.transfer as transfer
from cosmos_transfer1.utils import log

import gradio as gr


class CosmosTransferGradioApp:
    def __init__(self, checkpoint_dir="/mnt/pvc/cosmos-transfer1/", output_dir="/mnt/pvc/gradio/cosmos-transfer1"):
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    def create_controlnet_spec(self, control_types, vis_weight, edge_weight, depth_weight, seg_weight, keypoint_weight):
        """Create a controlnet spec dictionary from UI inputs"""
        control_inputs = {}

        if "Visual" in control_types and vis_weight > 0:
            control_inputs["vis"] = {"control_weight": vis_weight}

        if "Edge" in control_types and edge_weight > 0:
            control_inputs["edge"] = {"control_weight": edge_weight}

        if "Depth" in control_types and depth_weight > 0:
            control_inputs["depth"] = {"control_weight": depth_weight}

        if "Segmentation" in control_types and seg_weight > 0:
            control_inputs["seg"] = {"control_weight": seg_weight}

        if "Keypoint" in control_types and keypoint_weight > 0:
            control_inputs["keypoint"] = {"control_weight": keypoint_weight}

        return control_inputs

    def create_args_namespace(
        self,
        input_video,
        prompt,
        negative_prompt,
        control_inputs,
        guidance_scale,
        num_steps,
        seed,
        sigma_max,
        blur_strength,
        canny_threshold,
        output_folder,
    ):
        """Create an argparse.Namespace object that mimics command line arguments in transfer.py"""
        args = argparse.Namespace()

        # Required arguments
        args.checkpoint_dir = self.checkpoint_dir
        args.controlnet_specs = None  # We'll pass control_inputs directly

        # Video and prompt settings
        args.input_video_path = input_video
        args.prompt = prompt
        args.negative_prompt = negative_prompt
        args.video_save_folder = output_folder
        args.video_save_name = "output"

        # Generation parameters
        args.guidance = guidance_scale
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
        args.num_gpus = 1

        # Memory optimization
        args.offload_diffusion_transformer = True
        args.offload_text_encoder_model = True
        args.offload_guardrail_models = True
        args.upsample_prompt = False
        args.offload_prompt_upsampler = True

        # Batch processing (not used in this case)
        args.batch_input_path = None

        args.use_distilled = False

        return args

    def infer(
        self,
        input_video,
        prompt,
        negative_prompt,
        control_types,
        vis_weight,
        edge_weight,
        depth_weight,
        seg_weight,
        keypoint_weight,
        guidance_scale,
        num_steps,
        seed,
        sigma_max,
        blur_strength,
        canny_threshold,
    ) -> tuple[Optional[str], Optional[str]]:
        """Inference function for Gradio app

        - Takes arguments from Gradio UI
        - Reshapes them into an argparse.Namespace and controlnet_specs object (mimics transfer.py#parse_arguments())
        - Passes them to transfer.demo()
        - Returns the output video path and a success message
        - Returns None and an error message if the input video is not provided or if the generation fails
        """
        if not input_video:
            log.error("No input video provided")
            return None, "Please upload an input video"

        try:
            # Create control inputs
            control_inputs = self.create_controlnet_spec(
                control_types, vis_weight, edge_weight, depth_weight, seg_weight, keypoint_weight
            )

            if not control_inputs:
                return None, "Please select at least one control type with weight > 0"

            # Create unique output folder with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = os.path.join(self.output_dir, f"generation_{timestamp}")
            os.makedirs(output_folder, exist_ok=True)

            # Copy input video to output folder for reference
            input_filename = os.path.basename(input_video)
            input_copy_path = os.path.join(output_folder, f"input_{input_filename}")
            shutil.copy2(input_video, input_copy_path)

            # Create arguments namespace
            args = self.create_args_namespace(
                input_video=input_copy_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_inputs=control_inputs,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                seed=seed,
                sigma_max=sigma_max,
                blur_strength=blur_strength,
                canny_threshold=canny_threshold,
                output_folder=output_folder,
            )

            # Run the existing demo function
            transfer.demo(args, control_inputs)

            # Check if output was generated
            output_path = os.path.join(output_folder, "output.mp4")
            if os.path.exists(output_path):
                # Read the generated prompt
                prompt_path = os.path.join(output_folder, "output.txt")
                final_prompt = prompt
                if os.path.exists(prompt_path):
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        final_prompt = f.read().strip()

                return (
                    output_path,
                    f"Video generated successfully!\nOutput saved to: {output_folder}\nFinal prompt: {final_prompt}",
                )
            else:
                return None, f"Generation failed - no output video was created\nCheck folder: {output_folder}"

        except Exception as e:
            return None, f"Error during generation: {str(e)}"


def create_gradio_interface():
    app = CosmosTransferGradioApp()

    with gr.Blocks(title="Cosmos-Transfer1 Video Generation", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Cosmos-Transfer1: World Generation with Adaptive Multimodal Control")
        gr.Markdown("Upload a video and configure controls to generate a new video with the Cosmos-Transfer1 model.")
        gr.Markdown(f"**Output Directory**: {app.output_dir}")

        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                input_video = gr.Video(
                    label="Input Video",
                    height=300,
                    # Configure file upload settings
                    interactive=True,
                )

                prompt = gr.Textbox(
                    label="Prompt",
                    value="The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.",
                    lines=4,
                )

                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all.",
                    lines=3,
                )

                # Control selection
                gr.Markdown("### Control Types")

                with gr.Row():
                    with gr.Column(scale=1):
                        vis_enable = gr.Checkbox(label="Visual", value=False)
                    with gr.Column(scale=3):
                        vis_weight = gr.Slider(0, 1, value=0.5, step=0.1, label="Visual Weight", interactive=False)

                with gr.Row():
                    with gr.Column(scale=1):
                        edge_enable = gr.Checkbox(label="Edge", value=True)
                    with gr.Column(scale=3):
                        edge_weight = gr.Slider(0, 1, value=1.0, step=0.1, label="Edge Weight", interactive=True)

                with gr.Row():
                    with gr.Column(scale=1):
                        depth_enable = gr.Checkbox(label="Depth", value=False)
                    with gr.Column(scale=3):
                        depth_weight = gr.Slider(0, 1, value=0.5, step=0.1, label="Depth Weight", interactive=False)

                with gr.Row():
                    with gr.Column(scale=1):
                        seg_enable = gr.Checkbox(label="Segmentation", value=False)
                    with gr.Column(scale=3):
                        seg_weight = gr.Slider(
                            0, 1, value=0.5, step=0.1, label="Segmentation Weight", interactive=False
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        keypoint_enable = gr.Checkbox(label="Keypoint", value=False)
                    with gr.Column(scale=3):
                        keypoint_weight = gr.Slider(
                            0, 1, value=0.5, step=0.1, label="Keypoint Weight", interactive=False
                        )

                # Add interactivity to enable/disable sliders based on checkboxes
                vis_enable.change(fn=lambda x: gr.update(interactive=x), inputs=vis_enable, outputs=vis_weight)
                edge_enable.change(fn=lambda x: gr.update(interactive=x), inputs=edge_enable, outputs=edge_weight)
                depth_enable.change(fn=lambda x: gr.update(interactive=x), inputs=depth_enable, outputs=depth_weight)
                seg_enable.change(fn=lambda x: gr.update(interactive=x), inputs=seg_enable, outputs=seg_weight)
                keypoint_enable.change(
                    fn=lambda x: gr.update(interactive=x), inputs=keypoint_enable, outputs=keypoint_weight
                )

                # Advanced settings
                with gr.Accordion("Advanced Settings", open=False):
                    guidance_scale = gr.Slider(1, 15, value=7.0, step=0.5, label="Guidance Scale")
                    num_steps = gr.Slider(10, 50, value=35, step=1, label="Number of Steps")
                    seed = gr.Number(value=1, label="Seed", precision=0)
                    sigma_max = gr.Slider(0, 80, value=70.0, step=1, label="Sigma Max")

                    blur_strength = gr.Dropdown(
                        choices=["very_low", "low", "medium", "high", "very_high"],
                        value="medium",
                        label="Blur Strength",
                    )

                    canny_threshold = gr.Dropdown(
                        choices=["very_low", "low", "medium", "high", "very_high"],
                        value="medium",
                        label="Canny Threshold",
                    )

                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Output
                output_video = gr.Video(label="Generated Video", height=400)
                status_text = gr.Textbox(label="Status", lines=5, interactive=False)

        # Event handler
        def infer_wrapper(
            input_video,
            prompt,
            negative_prompt,
            vis_enable,
            vis_weight,
            edge_enable,
            edge_weight,
            depth_enable,
            depth_weight,
            seg_enable,
            seg_weight,
            keypoint_enable,
            keypoint_weight,
            guidance_scale,
            num_steps,
            seed,
            sigma_max,
            blur_strength,
            canny_threshold,
        ):
            # Convert individual checkboxes to control_types list
            control_types = []
            if vis_enable:
                control_types.append("Visual")
            if edge_enable:
                control_types.append("Edge")
            if depth_enable:
                control_types.append("Depth")
            if seg_enable:
                control_types.append("Segmentation")
            if keypoint_enable:
                control_types.append("Keypoint")

            return app.infer(
                input_video,
                prompt,
                negative_prompt,
                control_types,
                vis_weight,
                edge_weight,
                depth_weight,
                seg_weight,
                keypoint_weight,
                guidance_scale,
                num_steps,
                seed,
                sigma_max,
                blur_strength,
                canny_threshold,
            )

        generate_btn.click(
            fn=infer_wrapper,
            inputs=[
                input_video,
                prompt,
                negative_prompt,
                vis_enable,
                vis_weight,
                edge_enable,
                edge_weight,
                depth_enable,
                depth_weight,
                seg_enable,
                seg_weight,
                keypoint_enable,
                keypoint_weight,
                guidance_scale,
                num_steps,
                seed,
                sigma_max,
                blur_strength,
                canny_threshold,
            ],
            outputs=[output_video, status_text],
        )

        # Examples section
        gr.Markdown("## Tips for better results:")
        gr.Markdown(
            """
        - **Describe a single, captivating scene**: Focus on one scene to prevent unnecessary shot changes
        - **Use detailed prompts**: Rich descriptions lead to better quality outputs  
        - **Experiment with control weights**: Different combinations can yield different artistic effects
        - **Adjust sigma_max**: Lower values preserve more of the input video structure
        """
        )

        gr.Markdown("## File Storage:")
        gr.Markdown(
            f"""
        - **Input videos**: Temporarily stored in Gradio's cache, then copied to output folder
        - **Generated videos**: Saved to `{app.output_dir}/generation_YYYYMMDD_HHMMSS/`
        - **Output structure**: Each generation gets its own timestamped folder with input copy, output video, and prompt
        """
        )

    return interface


if __name__ == "__main__":
    # Check if checkpoints exist
    if not os.path.exists("checkpoints"):
        print("Error: checkpoints directory not found. Please download the model checkpoints first.")
        print("Run: python scripts/download_checkpoints.py --output_dir checkpoints/")
        exit(1)

    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=True,
        # Configure file upload limits
        max_file_size="500MB",  # Adjust as needed
        allowed_paths=["/mnt/pvc/gradio"],  # Allow access to output directory
    )
