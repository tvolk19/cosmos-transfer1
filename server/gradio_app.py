# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from cosmos_transfer1.utils import log
from server.deploy_config import Config
from server.model_server import ModelServer
from server.gradio_app_cli import GradioCLIApp
from server.gradio_interface import create_gradio_interface
from gradio_util import get_output_folder, get_outputs, create_worker_pipeline


class GradioApp:
    def __init__(self, cfg):
        if cfg.num_gpus == 1:
            self.pipeline, self.validator = create_worker_pipeline(cfg)
        else:
            self.pipeline = ModelServer(num_workers=cfg.num_gpus)
            _, self.validator = create_worker_pipeline(cfg, create_model=False)

    def infer(
        self,
        request_text,
    ):
        output_folder = get_output_folder(Config.output_dir)

        try:
            request_data = json.loads(request_text)
        except json.JSONDecodeError as e:
            return None, f"Error parsing request JSON: {e}\nPlease ensure your request is valid JSON."

        try:
            log.info(f"Model parameters: {json.dumps(request_data, indent=4)}")

            args_dict = self.validator.parse_and_validate(request_data)
            args_dict["output_dir"] = output_folder

            self.pipeline.infer(args_dict)

        except Exception as e:
            log.error(f"Error during inference: {e}")
            return None, f"Error: {e}"

        return get_outputs(output_folder)


if __name__ == "__main__":

    cfg = Config()
    log.info(f"Starting Gradio app with config: {str(cfg)}")

    if not os.path.exists(cfg.checkpoint_dir):
        print(f"Error: checkpoints directory {cfg.checkpoint_dir} not found.")
        exit(1)

    if cfg.use_cli:
        app = GradioCLIApp(num_workers=cfg.num_gpus, checkpoint_dir=cfg.checkpoint_dir)
    else:
        app = GradioApp(cfg)

    interface = create_gradio_interface(app.infer)

    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=True,
        max_file_size="500MB",
        allowed_paths=[cfg.output_dir, cfg.uploads_dir],
    )
