import os
import copy
import json
import logging as log


class BatchTransferPipeline(CosmosTransferPipeline):

    def generate_batch(self, cfg, control_inputs, prompts):
        """Generate videos in batch mode"""
        batch_size = cfg.batch_size if hasattr(cfg, "batch_size") else 1
        if any("upscale" in str(control_input) for control_input in control_inputs) and batch_size > 1:
            batch_size = 1
            log.info("Setting batch_size=1 as upscale does not support batch generation")

        os.makedirs(cfg.video_save_folder, exist_ok=True)

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start : batch_start + batch_size]
            self._process_batch(cfg, control_inputs, batch_prompts, batch_start)

    def _process_batch(self, cfg, control_inputs, batch_prompts, batch_start):
        """Process a batch of prompts"""
        actual_batch_size = len(batch_prompts)
        batch_prompt_texts = [p.get("prompt", None) for p in batch_prompts]
        batch_video_paths = [p.get("visual_input", None) for p in batch_prompts]

        batch_control_inputs = []
        for i, input_dict in enumerate(batch_prompts):
            current_prompt = input_dict.get("prompt", None)
            current_video_path = input_dict.get("visual_input", None)

            if cfg.batch_input_path:
                video_save_subfolder = os.path.join(cfg.video_save_folder, f"video_{batch_start+i}")
                os.makedirs(video_save_subfolder, exist_ok=True)
            else:
                video_save_subfolder = cfg.video_save_folder

            current_control_inputs = copy.deepcopy(control_inputs)
            if "control_overrides" in input_dict:
                for hint_key, override in input_dict["control_overrides"].items():
                    if hint_key in current_control_inputs:
                        current_control_inputs[hint_key].update(override)
                    else:
                        log.warning(f"Ignoring unknown control key in override: {hint_key}")

            # Run preprocessor
            log.info("Running preprocessor")
            assert self.preprocessors is not None, "Preprocessors not initialized"
            self.preprocessors(
                current_video_path,
                current_prompt,
                current_control_inputs,
                video_save_subfolder,
                cfg.regional_prompts if hasattr(cfg, "regional_prompts") else None,
            )
            batch_control_inputs.append(current_control_inputs)

        # Handle regional prompts
        self._setup_regional_prompts(cfg)

        # Generate videos in batch
        batch_outputs = self.pipeline.generate(
            prompt=batch_prompt_texts,
            video_path=batch_video_paths,
            negative_prompt=cfg.negative_prompt,
            control_inputs=batch_control_inputs,
            save_folder=video_save_subfolder,
            batch_size=actual_batch_size,
        )

        if batch_outputs is None:
            log.critical("Guardrail blocked generation for entire batch.")
            return

        # Save outputs
        self._save_batch_outputs(cfg, batch_outputs, batch_start)

    def _setup_regional_prompts(self, cfg):
        """Setup regional prompts if configured"""
        regional_prompts = []
        region_definitions = []

        if hasattr(cfg, "regional_prompts") and cfg.regional_prompts:
            log.info(f"regional_prompts: {cfg.regional_prompts}")
            for regional_prompt in cfg.regional_prompts:
                regional_prompts.append(regional_prompt["prompt"])
                if "region_definitions_path" in regional_prompt:
                    log.info(f"region_definitions_path: {regional_prompt['region_definitions_path']}")
                    region_definition_path = regional_prompt["region_definitions_path"]
                    if isinstance(region_definition_path, str) and region_definition_path.endswith(".json"):
                        with open(region_definition_path, "r") as f:
                            region_definitions_json = json.load(f)
                        region_definitions.extend(region_definitions_json)
                    else:
                        region_definitions.append(region_definition_path)

        if hasattr(self.pipeline, "regional_prompts"):
            self.pipeline.regional_prompts = regional_prompts
        if hasattr(self.pipeline, "region_definitions"):
            self.pipeline.region_definitions = region_definitions

    def _save_batch_outputs(self, cfg, batch_outputs, batch_start):
        """Save batch outputs to files"""
        videos, final_prompts = batch_outputs

        for i, (video, prompt) in enumerate(zip(videos, final_prompts)):
            if cfg.batch_input_path:
                video_save_subfolder = os.path.join(cfg.video_save_folder, f"video_{batch_start+i}")
                video_save_path = os.path.join(video_save_subfolder, "output.mp4")
                prompt_save_path = os.path.join(video_save_subfolder, "prompt.txt")
            else:
                video_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.mp4")
                prompt_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.txt")

            if self.device_rank == 0:
                os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
                save_video(
                    video=video,
                    fps=cfg.fps,
                    H=video.shape[1],
                    W=video.shape[2],
                    video_save_quality=5,
                    video_save_path=video_save_path,
                )

                with open(prompt_save_path, "wb") as f:
                    f.write(prompt.encode("utf-8"))

                log.info(f"Saved video to {video_save_path}")
                log.info(f"Saved prompt to {prompt_save_path}")
