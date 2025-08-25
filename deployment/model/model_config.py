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

import os
from dataclasses import dataclass, field
from typing import Dict
import json


default_request_vis = json.dumps(
    {
        "prompt_path": "assets/robot_example/robot_prompt.txt",
        "video_path": "assets/robot_example/robot_input.mp4",
        "control_weight": 1.0,
        "vis": {"control_path": "assets/robot_example/vis/robot_vis.mp4"},
    },
    indent=2,
)

default_request_depth = json.dumps(
    {
        "prompt_path": "assets/robot_example/robot_prompt.txt",
        "video_path": "assets/robot_example/robot_input.mp4",
        "control_weight": 1.0,
        "depth": {"control_path": "assets/robot_example/depth/robot_depth.mp4"},
    },
    indent=2,
)

default_request_edge = json.dumps(
    {
        "prompt_path": "assets/robot_example/robot_prompt.txt",
        "video_path": "assets/robot_example/robot_input.mp4",
        "control_weight": 1.0,
        "edge": {"control_path": "assets/robot_example/edge/robot_edge.mp4", "preset_edge_threshold": "medium"},
    },
    indent=2,
)

default_request_seg = json.dumps(
    {
        "prompt_path": "assets/robot_example/robot_prompt.txt",
        "video_path": "assets/robot_example/robot_input.mp4",
        "control_weight": 1.0,
        "seg": {"control_path": "assets/robot_example/seg/robot_seg.mp4"},
    },
    indent=2,
)

help_text_vis = """
                    ### Generation Parameters:
                    - `prompt` (string): Text description of desired output (default: empty string)
                    - `negative_prompt` (string): What to avoid in generation (default: predefined negative prompt)

                    """
help_text_depth = help_text_vis
help_text_edge = help_text_vis
help_text_seg = help_text_vis


@dataclass
class Config:

    header: Dict[str, str] = field(
        default_factory=lambda: {
            "vis": "Cosmos-Transfer2 Blur Transfer",
            "depth": "Cosmos-Transfer2 Depth Transfer",
            "edge": "Cosmos-Transfer2 Edge Transfer",
            "seg": "Cosmos-Transfer2 Segmentation Transfer",
        }
    )

    help_text: Dict[str, str] = field(
        default_factory=lambda: {
            "vis": help_text_vis,
            "depth": help_text_depth,
            "edge": help_text_edge,
            "seg": help_text_seg,
        }
    )

    default_request: Dict[str, str] = field(
        default_factory=lambda: {
            "vis": default_request_vis,
            "depth": default_request_depth,
            "edge": default_request_edge,
            "seg": default_request_seg,
        }
    )
