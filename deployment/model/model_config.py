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


default_request_transfer = json.dumps(
    {
        "input_video_path": "assets/example1_input_video.mp4",
        "prompt": "The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.",
        "negative_prompt": "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all.",
        "guidance": 7.0,
        "num_steps": 35,
        "seed": 1,
        "sigma_max": 70.0,
        "blur_strength": "medium",
        "canny_threshold": "medium",
        "edge": {"control_weight": 1.0},
    },
    indent=2,
)

default_request_transfer_av = json.dumps(
    {
        "prompt": "The video is captured from a camera mounted on a car. The camera is facing forward. The video showcases a scenic golden-hour drive through a suburban area, bathed in the warm, golden hues of the setting sun. The dashboard camera captures the play of light and shadow as the sun’s rays filter through the trees, casting elongated patterns onto the road. The streetlights remain off, as the golden glow of the late afternoon sun provides ample illumination. The two-lane road appears to shimmer under the soft light, while the concrete barrier on the left side of the road reflects subtle warm tones. The stone wall on the right, adorned with lush greenery, stands out vibrantly under the golden light, with the palm trees swaying gently in the evening breeze. Several parked vehicles, including white sedans and vans, are seen on the left side of the road, their surfaces reflecting the amber hues of the sunset. The trees, now highlighted in a golden halo, cast intricate shadows onto the pavement. Further ahead, houses with red-tiled roofs glow warmly in the fading light, standing out against the sky, which transitions from deep orange to soft pastel blue. As the vehicle continues, a white sedan is seen driving in the same lane, while a black sedan and a white van move further ahead. The road markings are crisp, and the entire setting radiates a peaceful, almost cinematic beauty. The golden light, combined with the quiet suburban landscape, creates an atmosphere of tranquility and warmth, making for a mesmerizing and soothing drive.",
        "sigma_max": 80,
        "hdmap": {"control_weight": 0.3, "input_control": "assets/sample_av_multi_control_input_hdmap.mp4"},
        "lidar": {"control_weight": 0.7, "input_control": "assets/sample_av_multi_control_input_lidar.mp4"},
    },
    indent=2,
)

help_text_transfer = """
                    ### Generation Parameters:
                    - `prompt` (string): Text description of desired output (default: empty string)
                    - `negative_prompt` (string): What to avoid in generation (default: predefined negative prompt)

                    """
help_text_transfer_av = help_text_transfer


@dataclass
class Config:

    header: Dict[str, str] = field(
        default_factory=lambda: {
            "transfer": "Cosmos-Transfer1",
            "transfer_av": "Cosmos-Transfer AV",
        }
    )

    help_text: Dict[str, str] = field(
        default_factory=lambda: {
            "transfer": help_text_transfer,
            "transfer_av": help_text_transfer_av,
        }
    )

    default_request: Dict[str, str] = field(
        default_factory=lambda: {
            "transfer": default_request_transfer,
            "transfer_av": default_request_transfer_av,
        }
    )
