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
from io import BytesIO
from typing import Dict, List
from pathlib import Path
import imageio
import numpy as np
import cv2
from cosmos_transfer1.utils import log


def read_prompts_from_file(prompt_file: str) -> List[Dict[str, str]]:
    """Read prompts from a JSONL file where each line is a dict with 'prompt' key and optionally 'visual_input' key.

    Args:
        prompt_file (str): Path to JSONL file containing prompts

    Returns:
        List[Dict[str, str]]: List of prompt dictionaries
    """
    prompts = []
    with open(prompt_file, "r") as f:
        for line in f:
            prompt_dict = json.loads(line.strip())
            prompts.append(prompt_dict)
    return prompts


def save_video(video, fps, H, W, video_save_quality, video_save_path):
    """Save video frames to file.

    Args:
        grid (np.ndarray): Video frames array [T,H,W,C]
        fps (int): Frames per second
        H (int): Frame height
        W (int): Frame width
        video_save_quality (int): Video encoding quality (0-10)
        video_save_path (str): Output video file path
    """
    kwargs = {
        "fps": fps,
        "quality": video_save_quality,
        "macro_block_size": 1,
        "ffmpeg_params": ["-s", f"{W}x{H}"],
        "output_params": ["-f", "mp4"],
    }
    imageio.mimsave(video_save_path, video, "mp4", **kwargs)


def load_from_fileobj(filepath: str, format: str = "mp4", mode: str = "rgb", **kwargs):
    """
    Load video from a file-like object using imageio with specified format and color mode.

    Parameters:
        file (IO[bytes]): A file-like object containing video data.
        format (str): Format of the video file (default 'mp4').
        mode (str): Color mode of the video, 'rgb' or 'gray' (default 'rgb').

    Returns:
        tuple: A tuple containing an array of video frames and metadata about the video.
    """
    with open(filepath, "rb") as f:
        value = f.read()
    with BytesIO(value) as f:
        f.seek(0)
        video_reader = imageio.get_reader(f, format, **kwargs)

        video_frames = []
        for frame in video_reader:
            if mode == "gray":
                import cv2  # Convert frame to grayscale if mode is gray

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = np.expand_dims(frame, axis=2)  # Keep frame dimensions consistent
            video_frames.append(frame)

    return np.array(video_frames), video_reader.get_meta_data()


def _get_codec_name(fourcc):
    """Convert fourcc code to readable codec name."""
    try:
        # Convert fourcc to 4-character string
        fourcc_str = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
        return fourcc_str.strip()
    except Exception:
        return "Unknown"


def validate_input_video(input_video_path):
    """Validate input video file for compatibility with the transfer pipeline.

    Performs comprehensive validation including:
    - File existence and accessibility
    - Video format and codec support
    - Basic video properties (dimensions, frame count, duration)
    - Frame rate validation
    - Support for multiple color formats: grayscale, RGB/BGR, RGBA/BGRA

    Args:
        input_video_path (str): Path to the input video file

    Returns:
        dict: Video properties including fps, frame_count, width, height, duration,
                channels, format_type, codec, and file_size

    Raises:
        ValueError: If video is invalid or incompatible
        FileNotFoundError: If video file doesn't exist
    """
    if not input_video_path:
        raise ValueError("Input video path cannot be None or empty")

    # Check file existence
    video_path = Path(input_video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")

    if not video_path.is_file():
        raise ValueError(f"Input video path is not a file: {input_video_path}")

    # Open video with OpenCV
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(
            f"Cannot open video file: {input_video_path}. " "File may be corrupted or in an unsupported format."
        )

    try:
        # Get basic video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)

        # Validate basic properties
        if fps <= 0:
            raise ValueError(f"Invalid frame rate: {fps}. Video may be corrupted.")

        if frame_count <= 0:
            raise ValueError(f"Invalid frame count: {frame_count}. Video may be empty or corrupted.")

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid video dimensions: {width}x{height}")

        # Calculate duration
        duration = frame_count / fps if fps > 0 else 0

        # Validate minimum requirements
        if width < 64 or height < 64:
            raise ValueError(f"Video resolution too small: {width}x{height}. " "Minimum resolution is 64x64 pixels.")

        if duration < 0.1:  # Less than 100ms
            raise ValueError(f"Video too short: {duration:.2f} seconds. " "Minimum duration is 0.1 seconds.")

        if duration > 300:  # More than 5 minutes
            log.warning(
                f"Video is very long: {duration:.2f} seconds. "
                "Processing may take a significant amount of time and memory."
            )

        # Check if we can read at least the first frame
        ret, frame = cap.read()
        if not ret or frame is None:
            raise ValueError("Cannot read video frames. Video may be corrupted.")

        # Validate frame properties - support RGB/BGR, RGBA/BGRA, and grayscale
        if len(frame.shape) == 2:
            # Grayscale video (height, width)
            log.debug(f"Detected grayscale video format: {frame.shape}")
        elif len(frame.shape) == 3:
            channels = frame.shape[2]
            if channels == 3:
                # RGB/BGR video
                log.debug(f"Detected RGB/BGR video format: {frame.shape}")
            elif channels == 4:
                # RGBA/BGRA video with alpha channel
                log.debug(f"Detected RGBA/BGRA video format with alpha channel: {frame.shape}")
            else:
                raise ValueError(
                    f"Unsupported number of channels: {channels}. "
                    f"Supported formats: grayscale (1 channel), RGB/BGR (3 channels), "
                    f"or RGBA/BGRA (4 channels). Got shape: {frame.shape}"
                )
        else:
            raise ValueError(
                f"Invalid frame format. Expected 2D (grayscale) or 3D (color) array, "
                f"got {len(frame.shape)}D array with shape: {frame.shape}"
            )

        # Additional checks for very high resolution videos
        if width > 4096 or height > 4096:
            log.warning(
                f"Very high resolution video: {width}x{height}. "
                "This may require significant memory and processing time."
            )

        # Check frame rate range
        if fps > 120:
            log.warning(f"Very high frame rate: {fps} fps. Consider reducing for better performance.")
        elif fps < 1:
            log.warning(f"Very low frame rate: {fps} fps. This may affect output quality.")
        elif fps != 24 and fps != 30 and fps != 60:
            log.warning(f"Uncommon frame rate: {fps} fps. " "Ensure your processing pipeline supports this frame rate.")

        # Determine format info from frame
        if len(frame.shape) == 2:
            channels = 1
            format_type = "grayscale"
        else:
            channels = frame.shape[2]
            if channels == 3:
                format_type = "RGB/BGR"
            elif channels == 4:
                format_type = "RGBA/BGRA"
            else:
                format_type = f"{channels}-channel"

        video_info = {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "fourcc": fourcc,
            "file_size": video_path.stat().st_size,
            "codec": _get_codec_name(fourcc),
            "channels": channels,
            "format_type": format_type,
        }

        log.info(
            f"Video validation successful: {input_video_path} {width}x{height}, {fps:.2f} fps, {format_type}, "
            f"{frame_count} frames, {duration:.2f}s duration"
        )

        return video_info

    finally:
        cap.release()
