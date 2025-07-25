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

from cosmos_transfer1.diffusion.inference.dummy_pipeline import create_dummy_pipeline
from server.model_server import ModelServer
from cosmos_transfer1.utils import log


def create_worker_pipeline(cfg, create_model=True):
    module = __import__(cfg.factory_module, fromlist=[cfg.factory_function])
    factory_function = getattr(module, cfg.factory_function)
    log.info(f"initializing model using {cfg.factory_module}.{cfg.factory_function}")
    return factory_function(cfg, create_model=create_model)


def create_pipeline_IPC(cfg):
    pipeline = ModelServer(num_workers=cfg.num_gpus)
    _, validator = create_worker_pipeline(cfg, create_model=False)
    return pipeline, validator


def create_pipeline(cfg):

    if cfg.num_gpus == 0:
        pipeline, validator = create_dummy_pipeline(cfg)
    elif cfg.num_gpus == 1:
        pipeline, validator = create_worker_pipeline(cfg)
    else:
        pipeline, validator = create_pipeline_IPC(cfg)

    return pipeline, validator
