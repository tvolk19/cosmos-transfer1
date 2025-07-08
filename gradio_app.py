import os
import sys
import json
import gradio as gr
import subprocess
import logging
import shutil
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
import time
import tempfile

theme = gr.themes.Soft()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


class TransferApp:
    def __init__(self):
        self.checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/mnt/pvc/checkpoints")
        self.num_gpus = int(os.getenv("NUM_GPU", "1"))  # Get NUM_GPU from environment, default to 1
        self.base_config = {
            "prompt": "",
            "input_video_path": "",
            "vis": {"control_weight": 0.0},
            "depth": {"control_weight": 0.0},
            "seg": {"control_weight": 0.0},
            "edge": {"control_weight": 0.0},
            "keypoint": {"control_weight": 0.0},
            "hdmap": {"control_weight": 0.0},
            "lidar": {"control_weight": 0.0},
        }

    def handle_video_removal(self, control_type: str, config: Dict[str, Any]) -> None:
        """Handle explicit video removal for a specific control type"""
        if control_type in config:
            if "input_control" in config[control_type]:
                logger.info(f"Removing {control_type} video input")
                del config[control_type]["input_control"]

    def update_config(
        self,
        prompt: str,
        input_video: Optional[str],
        vis_video: Optional[str],
        depth_video: Optional[str],
        seg_video: Optional[str],
        edge_video: Optional[str],
        keypoint_video: Optional[str],
        hdmap_video: Optional[str],
        lidar_video: Optional[str],
        config_json: str,
        enable_regional_prompts: bool,
        regional_prompts: List[Dict[str, str]],
        guidance_scale: float,
        num_steps: int,
        seed: int,
        sigma_max: float,
        blur_strength: str,
        canny_threshold: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Update the configuration with uploaded video paths and advanced settings"""
        try:
            # Try to parse the user-edited JSON
            try:
                config = json.loads(config_json)
            except json.JSONDecodeError as e:
                # If there's a JSON error, try to fix common issues
                try:
                    # Remove trailing commas
                    fixed_json = config_json.replace(",\n}", "\n}").replace(",\n]", "\n]")
                    # Try parsing again
                    config = json.loads(fixed_json)
                except json.JSONDecodeError:
                    # If still can't parse, log the error and return the original JSON
                    logger.error(f"JSON decode error: {str(e)}")
                    return json.loads(json.dumps(self.base_config)), json.loads(json.dumps(self.base_config))

            # Update video paths only if control weight is not 0
            if input_video:
                config["input_video_path"] = input_video
                logger.info("Updated input video path")
            else:
                if "input_video_path" in config and config["input_video_path"]:
                    logger.info("Removing input video path")
                    config["input_video_path"] = ""

            # Helper function to check control weight and update input
            def update_control_input(control_type: str, video_path: Optional[str]) -> None:
                if control_type in config:
                    control_weight = config[control_type].get("control_weight", 0)
                    try:
                        control_weight = float(control_weight)
                    except (ValueError, TypeError):
                        control_weight = 0

                    if control_weight == 0:
                        # Remove the control type entirely if weight is 0
                        logger.info(f"Removing {control_type} control due to zero weight")
                        del config[control_type]
                    else:
                        if video_path:
                            config[control_type]["input_control"] = video_path
                            logger.info(f"Updated {control_type} video path")
                        else:
                            # Explicitly handle video removal
                            self.handle_video_removal(control_type, config)

            # Update control inputs based on their weights
            update_control_input("vis", vis_video)
            update_control_input("depth", depth_video)
            update_control_input("seg", seg_video)
            update_control_input("edge", edge_video)
            update_control_input("keypoint", keypoint_video)
            update_control_input("hdmap", hdmap_video)
            update_control_input("lidar", lidar_video)

            # Update prompt
            config["prompt"] = prompt

            # Add regional prompts only if enabled
            if enable_regional_prompts and regional_prompts:
                config["regional_prompts"] = regional_prompts
                logger.info(f"Added {len(regional_prompts)} regional prompts to config")
            elif "regional_prompts" in config:
                logger.info("Removing regional prompts from config")
                del config["regional_prompts"]

            # Create a copy of the config for display (without advanced settings)
            display_config = config.copy()

            # Add advanced settings to the actual config used for generation
            config["guidance"] = guidance_scale
            config["num_steps"] = num_steps
            config["seed"] = seed
            config["sigma_max"] = sigma_max
            config["blur_strength"] = blur_strength
            config["canny_threshold"] = canny_threshold

            return display_config, config

        except Exception as e:
            logger.error(f"Error updating config: {str(e)}")
            return json.loads(json.dumps(self.base_config)), json.loads(json.dumps(self.base_config))

    def generate_video(
        self,
        prompt: str,
        input_video: Optional[str],
        vis_video: Optional[str],
        depth_video: Optional[str],
        seg_video: Optional[str],
        edge_video: Optional[str],
        keypoint_video: Optional[str],
        hdmap_video: Optional[str],
        lidar_video: Optional[str],
        config_json: str,
        output_folder: str,
        guidance_scale: float,
        num_steps: int,
        seed: int,
        sigma_max: float,
        blur_strength: str,
        canny_threshold: str,
    ) -> tuple[Optional[str], str, Optional[str], str]:
        """Generate video using transfer.py with advanced settings"""
        try:
            # Validate sigma_max and input video requirement
            if sigma_max < 80 and not input_video:
                error_msg = "Set sigma_max to 80 or higher if no input video is provided. Change the sigma_max in the Advanced Settings panel."
                logger.error(error_msg)
                return None, error_msg, None, error_msg

            # Start timing
            start_time = time.time()

            # Create a timestamped subfolder for this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_folder = os.path.join(output_folder, f"run_{timestamp}")
            os.makedirs(run_folder, exist_ok=True)

            # Parse regional prompts from config_json to get the actual regional prompts
            try:
                config_data = json.loads(config_json)
                actual_regional_prompts = config_data.get("regional_prompts", [])
                logger.info(
                    f"Found {len(actual_regional_prompts)} regional prompts in config: {actual_regional_prompts}"
                )
            except json.JSONDecodeError:
                actual_regional_prompts = []
                logger.warning("Failed to parse config_json, no regional prompts will be used")

            # Update configuration with video paths and advanced settings
            display_config, final_config = self.update_config(
                prompt=prompt,
                input_video=input_video,
                vis_video=vis_video,
                depth_video=depth_video,
                seg_video=seg_video,
                edge_video=edge_video,
                keypoint_video=keypoint_video,
                hdmap_video=hdmap_video,
                lidar_video=lidar_video,
                config_json=config_json,
                enable_regional_prompts=len(actual_regional_prompts) > 0,
                regional_prompts=actual_regional_prompts,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                seed=seed,
                sigma_max=sigma_max,
                blur_strength=blur_strength,
                canny_threshold=canny_threshold,
            )

            # Save the final configuration
            config_path = os.path.join(run_folder, "config.json")
            logger.info(f"Saving final config to {config_path}")
            logger.info(f"Final config contains regional_prompts: {'regional_prompts' in final_config}")
            if "regional_prompts" in final_config:
                logger.info(f"Regional prompts in final config: {final_config['regional_prompts']}")
            with open(config_path, "w") as f:
                json.dump(final_config, f, indent=4)

            # Build command with PYTHONPATH set
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

            # Check if hdmap or lidar control types are present
            has_av_controls = any(ctrl_type in final_config for ctrl_type in ["hdmap", "lidar"])

            cmd = [
                "torchrun",
                "--nproc_per_node=" + str(self.num_gpus),
                "--nnodes=1",
                "--node_rank=0",
                "cosmos_transfer1/diffusion/inference/transfer.py",
                "--checkpoint_dir",
                self.checkpoint_dir,
                "--video_save_folder",
                run_folder,
                "--controlnet_specs",
                config_path,
                "--guidance",
                str(guidance_scale),
                "--num_steps",
                str(num_steps),
                "--seed",
                str(seed),
                "--sigma_max",
                str(sigma_max),
                "--blur_strength",
                blur_strength,
                "--canny_threshold",
                canny_threshold,
                "--offload_text_encoder_model",
                # "--offload_guardrail_models",
                "--num_gpus",
                str(self.num_gpus),  # Add num_gpus parameter
            ]

            # Add --is_av_sample flag if hdmap or lidar controls are present
            if has_av_controls:
                cmd.append("--is_av_sample")

            logger.info(f"Running command: {' '.join(cmd)}")

            # Create a list to store terminal output
            terminal_output = []

            # Run the command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=env,
                bufsize=1,  # Line buffered
            )

            # Stream stdout and stderr in real-time
            def stream_output(pipe, is_error=False):
                for line in pipe:
                    line = line.strip()
                    if line:  # Only add non-empty lines
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if is_error:
                            formatted_line = f"[{timestamp}] ERROR: {line}"
                            logger.error(line)
                        else:
                            formatted_line = f"[{timestamp}] INFO: {line}"
                            logger.info(line)
                        terminal_output.append(formatted_line)

            # Start threads to stream output
            stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, False))
            stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, True))

            stdout_thread.start()
            stderr_thread.start()

            # Wait for process to complete
            process.wait()

            # Wait for output threads to complete
            stdout_thread.join()
            stderr_thread.join()

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            elapsed_minutes = int(elapsed_time // 60)
            elapsed_seconds = int(elapsed_time % 60)

            if process.returncode == 0:
                output_video = os.path.join(run_folder, "output.mp4")
                if os.path.exists(output_video):
                    status_msg = f"Generation completed successfully in {elapsed_minutes}m {elapsed_seconds}s"
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    terminal_output.append(f"[{timestamp}] INFO: {status_msg}")
                    logger.info(status_msg)
                    logger.info(f"Output video exists at: {output_video}")
                    terminal_output.append(f"[{timestamp}] INFO: Output video exists at: {output_video}")

                    # Create zip file of only the current run's content
                    zip_filename = f"transfer_output_{timestamp}.zip"
                    zip_path = os.path.join(output_folder, zip_filename)

                    try:
                        # Create zip file
                        shutil.make_archive(
                            os.path.splitext(zip_path)[0],  # Remove .zip extension as make_archive adds it
                            "zip",
                            run_folder,
                        )
                        logger.info(f"Created zip file at: {zip_path}")
                        terminal_output.append(f"[{timestamp}] INFO: Created zip file at: {zip_path}")

                        if not os.path.exists(zip_path):
                            error_msg = f"Zip file was not created at: {zip_path}"
                            logger.error(error_msg)
                            terminal_output.append(f"[{timestamp}] ERROR: {error_msg}")
                            return output_video, status_msg, None, "\n".join(terminal_output)

                        return output_video, status_msg, zip_path, "\n".join(terminal_output)
                    except Exception as e:
                        import traceback

                        error_msg = f"Error creating zip file:\n{traceback.format_exc()}"
                        logger.error(error_msg, exc_info=True)
                        terminal_output.append(f"[{timestamp}] ERROR: {error_msg}")
                        return output_video, status_msg, None, "\n".join(terminal_output)
                else:
                    error_msg = f"Error: Output video not found at {output_video}"
                    logger.error(error_msg)
                    terminal_output.append(f"[{timestamp}] ERROR: {error_msg}")
                    return None, error_msg, None, "\n".join(terminal_output)
            else:
                # Get the complete stderr output
                stderr_lines = []
                for line in process.stderr:
                    stderr_lines.append(line.strip())
                error_output = "\n".join(stderr_lines) if stderr_lines else "No error output available"
                # Show simpler message in status panel
                status_msg = "An error occurred during generation. Please check the terminal output for details."
                return None, status_msg, None, "\n".join(terminal_output)

        except Exception as e:
            import traceback

            error_msg = f"Error:\n{traceback.format_exc()}"
            logger.error(error_msg, exc_info=True)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Show simpler message in status panel
            status_msg = "An error occurred during generation. Please check the terminal output for details."
            return None, status_msg, None, f"[{timestamp}] ERROR: {error_msg}"


def create_app():
    app = TransferApp()

    with gr.Blocks(title="Cosmos Transfer", theme=theme) as demo:
        # Create a row for the header with icon and title
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("# Cosmos-Transfer1")

        # Hidden output folder
        output_folder = gr.Textbox(value="/mnt/pvc/gradio/cosmos-transfer1/output", visible=False)

        with gr.Row():
            with gr.Column():
                # Text inputs
                prompt = gr.Textbox(label="Prompt", placeholder="Describe the video you want to generate...", lines=3)

                # Main input video
                input_video = gr.Video(label="Input Video", sources=["upload"])

                # Controlnet inputs in collapsible panel
                with gr.Accordion("ControlNet Inputs", open=False):
                    vis_video = gr.Video(label="Visual Video", sources=["upload"])
                    depth_video = gr.Video(label="Depth Video", sources=["upload"])
                    seg_video = gr.Video(label="Segmentation Video", sources=["upload"])
                    edge_video = gr.Video(label="Edge Video", sources=["upload"])
                    keypoint_video = gr.Video(label="Keypoint Video", sources=["upload"])
                    hdmap_video = gr.Video(label="HD Map Video", sources=["upload"])
                    lidar_video = gr.Video(label="LiDAR Video", sources=["upload"])

                # Regional prompts panel
                with gr.Accordion("Regional Prompts", open=False):
                    gr.Markdown(
                        """
                    Add regional prompts for specific areas of the video.
                    Each region needs:
                    1. A prompt describing what should appear in that region
                    2. Region coordinates - either:
                       - **Coordinate list**: `[x1, y1, x2, y2]` where x1,y1 is top-left and x2,y2 is bottom-right (0-1)
                       - **File path**: Path to a JSON file containing the coordinates

                    **Important**: After adding regions, click "Apply" to update the configuration before generating!
                    """
                    )

                    enable_regional_prompts = gr.Checkbox(
                        label="Enable Regional Prompts", value=False, info="Toggle to enable/disable regional prompts"
                    )

                    regional_prompt = gr.Textbox(
                        label="Regional Prompt", placeholder="Describe what should appear in this region...", lines=2
                    )
                    mask_prompt = gr.Textbox(
                        label="Mask Prompt", placeholder="Describe the mask for this region...", lines=2, visible=False
                    )
                    region_coords = gr.Textbox(
                        label="Region Coordinates",
                        placeholder="[0.0, 0.0, 0.5, 1.0]",
                        info="Enter coordinates as [x1, y1, x2, y2]",
                    )
                    with gr.Row():
                        add_region_btn = gr.Button("Add Region", variant="primary")
                        clear_regions_btn = gr.Button("Clear All Regions", variant="secondary")

                    regions_list = gr.JSON(label="Current Regions", value=[], visible=True)

                    def add_region(prompt, mask_prompt, coords_str, current_regions, enabled):
                        if not enabled:
                            return current_regions, "", "", ""

                        if not prompt:
                            return current_regions, prompt, mask_prompt, coords_str

                        new_region = {"prompt": prompt}

                        # Only add mask_prompt if it's not empty
                        if mask_prompt and mask_prompt.strip():
                            new_region["mask_prompt"] = mask_prompt

                        # Only add region_definitions_path if coordinates are provided
                        if coords_str:
                            try:
                                # First try to parse as JSON (list of coordinates)
                                coords = json.loads(coords_str)
                                if isinstance(coords, list) and len(coords) == 4:
                                    # Use coordinates directly as list
                                    new_region["region_definitions_path"] = coords
                                    logger.info(f"Added coordinates as list: {coords}")
                                else:
                                    logger.warning("Coordinates must be a list of 4 numbers, skipping coordinates")
                            except json.JSONDecodeError:
                                # If JSON parsing fails, treat as file path
                                new_region["region_definitions_path"] = coords_str
                                logger.info(f"Added coordinates as file path: {coords_str}")

                        if current_regions is None:
                            current_regions = []

                        current_regions.append(new_region)
                        logger.info(f"Added region: {new_region}")

                        # Clear input fields after adding region
                        return current_regions, "", "", ""

                    add_region_btn.click(
                        fn=add_region,
                        inputs=[regional_prompt, mask_prompt, region_coords, regions_list, enable_regional_prompts],
                        outputs=[regions_list, regional_prompt, mask_prompt, region_coords],
                    )

                    def clear_all_regions():
                        logger.info("Cleared all regions")
                        return []

                    clear_regions_btn.click(fn=clear_all_regions, outputs=regions_list)

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

                # Configuration tips and documentation
                with gr.Accordion("Configuration Tips", open=False):
                    gr.Markdown(
                        """

                    <span style="color: red; font-weight: bold;">Make sure you click Apply when you are done with the configuration</span>

                    **Control Weights:**
                    - Each control type has a `control_weight` between 0 and 1
                    - Setting weight to 0 disables that control type or just remove the control type from the json
                    - Higher weights give more influence to that control
                    - Total of all weights should typically be 1.0

                    **Input Requirements:**
                    - When `sigma_max` < 80, an input video is required
                    - Each control type needs its corresponding input video
                    - Input videos are automatically added when uploaded

                    **Control Types:**
                    - **vis**: Visual control (RGB video)
                    - **depth**: Depth map control
                    - **seg**: Segmentation mask control
                    - **edge**: Edge detection control
                    - **keypoint**: Keypoint detection control
                    - **hdmap**: HD Map control
                    - **lidar**: LiDAR point cloud control
                    """
                    )

                # JSON configuration editor
                config_json = gr.Textbox(
                    label="Control Configuration",
                    value=json.dumps(app.base_config, indent=4),
                    lines=10,
                    info="Edit the JSON to control how different inputs influence the generation. See Configuration Tips for details.",
                )

                # Add Apply Configuration button with status
                with gr.Row():
                    apply_config_btn = gr.Button("Apply", variant="secondary")
                    config_status = gr.Textbox(label="Configuration Status", interactive=False, visible=True)

                generate_btn = gr.Button("Generate Video")

            with gr.Column():
                output_video = gr.Video(label="Generated Video", interactive=False)
                zip_file = gr.File(label="Download Files", interactive=False)
                status = gr.Textbox(label="Status", interactive=False)

                # Add collapsible terminal panel
                with gr.Accordion("Terminal Output", open=False):
                    terminal_output = gr.Textbox(label="Terminal", interactive=False, lines=20, show_copy_button=True)

        def update_final_config(
            prompt,
            input_video,
            vis_video,
            depth_video,
            seg_video,
            edge_video,
            keypoint_video,
            hdmap_video,
            lidar_video,
            config_json,
            enable_regional_prompts,
            regional_prompts,
            guidance_scale,
            num_steps,
            seed,
            sigma_max,
            blur_strength,
            canny_threshold,
        ):
            try:
                display_config, final_config = app.update_config(
                    prompt=prompt,
                    input_video=input_video,
                    vis_video=vis_video,
                    depth_video=depth_video,
                    seg_video=seg_video,
                    edge_video=edge_video,
                    keypoint_video=keypoint_video,
                    hdmap_video=hdmap_video,
                    lidar_video=lidar_video,
                    config_json=config_json,
                    enable_regional_prompts=enable_regional_prompts,
                    regional_prompts=regional_prompts,
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    seed=seed,
                    sigma_max=sigma_max,
                    blur_strength=blur_strength,
                    canny_threshold=canny_threshold,
                )

                # Generate status message about what was updated
                updates = []
                if input_video:
                    updates.append("Input video path")
                if vis_video:
                    updates.append("Visual control path")
                if depth_video:
                    updates.append("Depth control path")
                if seg_video:
                    updates.append("Segmentation control path")
                if edge_video:
                    updates.append("Edge control path")
                if keypoint_video:
                    updates.append("Keypoint control path")
                if hdmap_video:
                    updates.append("HD Map control path")
                if lidar_video:
                    updates.append("LiDAR control path")

                status_msg = "Configuration updated successfully"
                if updates:
                    status_msg += f" with: {', '.join(updates)}"

                # Parse and re-format the display config to ensure proper JSON formatting
                try:
                    if isinstance(display_config, str):
                        display_config = json.loads(display_config)
                    formatted_json = json.dumps(display_config, indent=4)
                    return formatted_json, status_msg
                except json.JSONDecodeError:
                    return config_json, status_msg
            except Exception as e:
                import traceback

                error_msg = f"Error updating configuration:\n{traceback.format_exc()}"
                logger.error(error_msg, exc_info=True)
                return config_json, error_msg

        # Update config only when Apply Configuration button is clicked
        apply_config_btn.click(
            fn=update_final_config,
            inputs=[
                prompt,
                input_video,
                vis_video,
                depth_video,
                seg_video,
                edge_video,
                keypoint_video,
                hdmap_video,
                lidar_video,
                config_json,
                enable_regional_prompts,
                regions_list,
                guidance_scale,
                num_steps,
                seed,
                sigma_max,
                blur_strength,
                canny_threshold,
            ],
            outputs=[config_json, config_status],
        )

        # Add clear event handlers for video components
        def handle_video_clear(control_type: str, *args):
            logger.info(f"Video cleared for {control_type}")
            return update_final_config(*args)

        # Set up clear event handlers for each video component
        input_video.clear(
            fn=lambda *args: handle_video_clear("input", *args),
            inputs=[
                prompt,
                input_video,
                vis_video,
                depth_video,
                seg_video,
                edge_video,
                keypoint_video,
                hdmap_video,
                lidar_video,
                config_json,
                guidance_scale,
                num_steps,
                seed,
                sigma_max,
                blur_strength,
                canny_threshold,
            ],
            outputs=[config_json],
        )

        vis_video.clear(
            fn=lambda *args: handle_video_clear("vis", *args),
            inputs=[
                prompt,
                input_video,
                vis_video,
                depth_video,
                seg_video,
                edge_video,
                keypoint_video,
                hdmap_video,
                lidar_video,
                config_json,
                guidance_scale,
                num_steps,
                seed,
                sigma_max,
                blur_strength,
                canny_threshold,
            ],
            outputs=[config_json],
        )

        depth_video.clear(
            fn=lambda *args: handle_video_clear("depth", *args),
            inputs=[
                prompt,
                input_video,
                vis_video,
                depth_video,
                seg_video,
                edge_video,
                keypoint_video,
                hdmap_video,
                lidar_video,
                config_json,
                guidance_scale,
                num_steps,
                seed,
                sigma_max,
                blur_strength,
                canny_threshold,
            ],
            outputs=[config_json],
        )

        seg_video.clear(
            fn=lambda *args: handle_video_clear("seg", *args),
            inputs=[
                prompt,
                input_video,
                vis_video,
                depth_video,
                seg_video,
                edge_video,
                keypoint_video,
                hdmap_video,
                lidar_video,
                config_json,
                guidance_scale,
                num_steps,
                seed,
                sigma_max,
                blur_strength,
                canny_threshold,
            ],
            outputs=[config_json],
        )

        edge_video.clear(
            fn=lambda *args: handle_video_clear("edge", *args),
            inputs=[
                prompt,
                input_video,
                vis_video,
                depth_video,
                seg_video,
                edge_video,
                keypoint_video,
                hdmap_video,
                lidar_video,
                config_json,
                guidance_scale,
                num_steps,
                seed,
                sigma_max,
                blur_strength,
                canny_threshold,
            ],
            outputs=[config_json],
        )

        keypoint_video.clear(
            fn=lambda *args: handle_video_clear("keypoint", *args),
            inputs=[
                prompt,
                input_video,
                vis_video,
                depth_video,
                seg_video,
                edge_video,
                keypoint_video,
                hdmap_video,
                lidar_video,
                config_json,
                guidance_scale,
                num_steps,
                seed,
                sigma_max,
                blur_strength,
                canny_threshold,
            ],
            outputs=[config_json],
        )

        hdmap_video.clear(
            fn=lambda *args: handle_video_clear("hdmap", *args),
            inputs=[
                prompt,
                input_video,
                vis_video,
                depth_video,
                seg_video,
                edge_video,
                keypoint_video,
                hdmap_video,
                lidar_video,
                config_json,
                guidance_scale,
                num_steps,
                seed,
                sigma_max,
                blur_strength,
                canny_threshold,
            ],
            outputs=[config_json],
        )

        lidar_video.clear(
            fn=lambda *args: handle_video_clear("lidar", *args),
            inputs=[
                prompt,
                input_video,
                vis_video,
                depth_video,
                seg_video,
                edge_video,
                keypoint_video,
                hdmap_video,
                lidar_video,
                config_json,
                guidance_scale,
                num_steps,
                seed,
                sigma_max,
                blur_strength,
                canny_threshold,
            ],
            outputs=[config_json],
        )

        # Generate button click handler with error handling
        def generate_with_error_handling(*args):
            try:
                # Clear previous outputs
                output_video.value = None
                zip_file.value = None
                status.value = "Starting generation..."
                terminal_output.value = ""  # Clear terminal output

                # Run generation
                video_path, status_msg, zip_path, terminal_output_text = app.generate_video(*args)

                # Update components with results
                if video_path:
                    output_video.value = video_path
                if zip_path:
                    zip_file.value = zip_path
                status.value = status_msg

                # Return all outputs in the correct order
                return [video_path, zip_path, status_msg, terminal_output_text]

            except Exception as e:
                import traceback

                error_msg = f"Error during generation:\n{traceback.format_exc()}"
                logger.error(error_msg, exc_info=True)
                status.value = error_msg
                terminal_output.value = error_msg
                return [None, None, error_msg, error_msg]

        # Generate button click handler
        generate_btn.click(
            fn=generate_with_error_handling,
            inputs=[
                prompt,
                input_video,
                vis_video,
                depth_video,
                seg_video,
                edge_video,
                keypoint_video,
                hdmap_video,
                lidar_video,
                config_json,
                output_folder,
                guidance_scale,
                num_steps,
                seed,
                sigma_max,
                blur_strength,
                canny_threshold,
            ],
            outputs=[output_video, zip_file, status, terminal_output],
            api_name="generate",
            show_progress=True,
        )

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=True,
        # Configure file upload limits
        # max_file_size="500MB",  # Adjust as needed
        allowed_paths=["/mnt/pvc/gradio"],  # Allow access to output directory
    )
