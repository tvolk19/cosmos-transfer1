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

from loguru import logger as log
from cosmos_gradio.gradio_app.gradio_app import GradioApp
from cosmos_gradio.gradio_app.gradio_ui import create_gradio_UI
from cosmos_gradio.deployment_env import DeploymentEnv
from deployment.model.model_config import Config as ModelConfig
from cosmos_transfer1.utils import log
from deployment.model.transfer_worker import TransferValidator, hint_keys, hint_keys_av

if __name__ == "__main__":
    model_cfg = ModelConfig()

    # configure server to use the transfer worker in the worker procs
    global_env = DeploymentEnv()

    log.info(f"Starting Gradio app with global env config: {global_env!s}")

    factory_funcs = {
        "transfer": "create_worker",
        "transfer_av": "create_worker_AV",
    }

    validators = {
        "transfer": TransferValidator(hint_keys=hint_keys),
        "transfer_av": TransferValidator(hint_keys=hint_keys_av),
    }

    app = GradioApp(
        num_gpus=global_env.num_gpus,
        validator=validators[global_env.model_name],
        factory_module="deployment.model.transfer_worker",
        factory_function=factory_funcs[global_env.model_name],
        output_dir=global_env.output_dir,
    )

    interface = create_gradio_UI(
        app.infer,
        header=model_cfg.header[global_env.model_name],
        default_request=model_cfg.default_request[global_env.model_name],
        help_text=model_cfg.help_text[global_env.model_name],
        uploads_dir=global_env.uploads_dir,
        output_dir=global_env.output_dir,
        log_file=global_env.log_file,
    )

    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=True,
        max_file_size="500MB",
        allowed_paths=[global_env.output_dir, global_env.uploads_dir],
    )
