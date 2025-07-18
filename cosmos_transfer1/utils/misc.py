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

from __future__ import annotations

import base64
import collections
import collections.abc
import functools
import json
import os
import random
import tempfile
import time
from contextlib import ContextDecorator
from typing import Any, Callable, List, Tuple, TypeVar

import cv2
import numpy as np
import termcolor
import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed._tensor.api import DTensor

from cosmos_transfer1.utils import distributed, log


import gc
import torch
import pynvml
import inspect


def get_gpu_mem():
    try:
        pynvml.nvmlInit()
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        return meminfo.used / meminfo.total * 100
    except pynvml.NVMLError as error:
        log.error(f"Failed to get GPU memory info: {error}")
        return 0


def print_gpu_memory(str=None):
    try:
        pynvml.nvmlInit()
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        log.info(
            f"{str}: {meminfo.used/1024/1024}/{meminfo.total/1024/1024}MiB used ({meminfo.free/1024/1024}MiB free)"
        )
    except pynvml.NVMLError as error:
        log.error(f"Failed to get GPU memory info: {error}")


def force_gc(msg=""):
    mem0 = get_gpu_mem()

    gc.collect()
    torch.cuda.empty_cache()

    mem1 = get_gpu_mem()
    log.info(f"GC: GPU memm: {mem0:.2f}% -> {mem1:.2f}% {msg}")


class MemoryTimer(ContextDecorator):

    def __init__(self, context: str = ""):
        self.context = context
        self.start_memory = 0
        self.end_memory = 0
        self.start_time = 0
        self.end_time = 0

    def __enter__(self):
        self.start_memory = get_gpu_mem()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_memory = get_gpu_mem()
        self.end_time = time.time()

        memory_diff = self.end_memory - self.start_memory
        time_elapsed = self.end_time - self.start_time

        context_str = f" [{self.context}]" if self.context else ""
        log.info(f"Memory{context_str}: {self.start_memory:.2f}% -> {self.end_memory:.2f}% (Δ{memory_diff:+.2f}%)")
        log.info(f"Time{context_str}: {time_elapsed:.4f} seconds")


class ScopeMemoryTimer:

    def __init__(self, context: str = ""):
        self.context = context
        self.end_memory = 0
        self.end_time = 0
        self.start_memory = get_gpu_mem()
        self.start_time = time.time()

    def __del__(self):
        self.end_memory = get_gpu_mem()
        self.end_time = time.time()

        memory_diff = self.end_memory - self.start_memory
        time_elapsed = self.end_time - self.start_time

        context_str = f" [{self.context}]" if self.context else ""
        log.info(
            f"Memory{context_str}: {self.start_memory:.2f}% -> {self.end_memory:.2f}% (Δ{memory_diff:+.2f}%)  Time{context_str}: {time_elapsed:.4f} seconds"
        )


if __name__ == "__main__":
    # Example usage of MemoryTimer
    with MemoryTimer("Example Context"):
        # Simulate some work
        time.sleep(1)

    # Example usage of ScopeMemoryTimer
    frame = inspect.currentframe()
    timer = ScopeMemoryTimer(f"Scope Example")
    time.sleep(1)
    # del timer  # This will trigger the memory and time logging upon deletion


def extract_video_frames(video_path, number_of_frames=2):
    cap = cv2.VideoCapture(video_path)
    frame_paths = []

    temp_dir = tempfile.gettempdir()
    for i in range(number_of_frames):  # Extract first two frames
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames

        temp_path = os.path.join(temp_dir, f"frame_{i+1}.png")
        cv2.imwrite(temp_path, frame)
        frame_paths.append(temp_path)

    cap.release()
    return frame_paths


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def to(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    memory_format: torch.memory_format = torch.preserve_format,
) -> Any:
    """Recursively cast data into the specified device, dtype, and/or memory_format.

    The input data can be a tensor, a list of tensors, a dict of tensors.
    See the documentation for torch.Tensor.to() for details.

    Args:
        data (Any): Input data.
        device (str | torch.device): GPU device (default: None).
        dtype (torch.dtype): data type (default: None).
        memory_format (torch.memory_format): memory organization format (default: torch.preserve_format).

    Returns:
        data (Any): Data cast to the specified device, dtype, and/or memory_format.
    """
    assert (
        device is not None or dtype is not None or memory_format is not None
    ), "at least one of device, dtype, memory_format should be specified"
    if isinstance(data, torch.Tensor):
        is_cpu = (isinstance(device, str) and device == "cpu") or (
            isinstance(device, torch.device) and device.type == "cpu"
        )
        data = data.to(
            device=device,
            dtype=dtype,
            memory_format=memory_format,
            non_blocking=(not is_cpu),
        )
        return data
    elif isinstance(data, collections.abc.Mapping):
        return type(data)({key: to(data[key], device=device, dtype=dtype, memory_format=memory_format) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return type(data)([to(elem, device=device, dtype=dtype, memory_format=memory_format) for elem in data])
    else:
        return data


def get_local_tensor_if_DTensor(tensor: torch.Tensor | DTensor) -> torch.tensor:
    if isinstance(tensor, DTensor):
        local = tensor.to_local()
        # As per PyTorch documentation, if the communication is not finished yet, we need to wait for it to finish
        # https://pytorch.org/docs/stable/distributed.tensor.html#torch.distributed.tensor.DTensor.to_local
        if isinstance(local, AsyncCollectiveTensor):
            return local.wait()
        else:
            return local
    return tensor


def serialize(data: Any) -> Any:
    """Serialize data by hierarchically traversing through iterables.

    Args:
        data (Any): Input data.

    Returns:
        data (Any): Serialized data.
    """
    if isinstance(data, collections.abc.Mapping):
        return type(data)({key: serialize(data[key]) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return type(data)([serialize(elem) for elem in data])
    else:
        try:
            json.dumps(data)
        except TypeError:
            data = str(data)
        return data


def print_environ_variables(env_vars: list[str]) -> None:
    """Print a specific list of environment variables.

    Args:
        env_vars (list[str]): List of specified environment variables.
    """
    for env_var in env_vars:
        if env_var in os.environ:
            log.info(f"Environment variable {Color.green(env_var)}: {Color.yellow(os.environ[env_var])}")
        else:
            log.warning(f"Environment variable {Color.green(env_var)} not set!")


def set_random_seed(seed: int, by_rank: bool = False) -> None:
    """Set random seed. This includes random, numpy, Pytorch.

    Args:
        seed (int): Random seed.
        by_rank (bool): if true, each GPU will use a different random seed.
    """
    if by_rank:
        seed += distributed.get_rank()
    log.info(f"Using random seed {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # sets seed on the current CPU & all GPUs


def arch_invariant_rand(
    shape: List[int] | Tuple[int], dtype: torch.dtype, device: str | torch.device, seed: int | None = None
):
    """Produce a GPU-architecture-invariant randomized Torch tensor.

    Args:
        shape (list or tuple of ints): Output tensor shape.
        dtype (torch.dtype): Output tensor type.
        device (torch.device): Device holding the output.
        seed (int): Optional randomization seed.

    Returns:
        tensor (torch.tensor): Randomly-generated tensor.
    """
    # Create a random number generator, optionally seeded
    rng = np.random.RandomState(seed)

    # # Generate random numbers using the generator
    random_array = rng.standard_normal(shape).astype(np.float32)  # Use standard_normal for normal distribution

    # Convert to torch tensor and return
    return torch.from_numpy(random_array).to(dtype=dtype, device=device)


T = TypeVar("T", bound=Callable[..., Any])


class timer(ContextDecorator):  # noqa: N801
    """Simple timer for timing the execution of code.

    It can be used as either a context manager or a function decorator. The timing result will be logged upon exit.

    Example:
        def func_a():
            time.sleep(1)
        with timer("func_a"):
            func_a()

        @timer("func_b)
        def func_b():
            time.sleep(1)
        func_b()
    """

    def __init__(self, context: str, debug: bool = False):
        self.context = context
        self.debug = debug

    def __enter__(self) -> None:
        self.tic = time.time()

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        time_spent = time.time() - self.tic
        if self.debug:
            log.debug(f"Time spent on {self.context}: {time_spent:.4f} seconds")
        else:
            log.debug(f"Time spent on {self.context}: {time_spent:.4f} seconds")

    def __call__(self, func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN202
            tic = time.time()
            result = func(*args, **kwargs)
            time_spent = time.time() - tic
            if self.debug:
                log.debug(f"Time spent on {self.context}: {time_spent:.4f} seconds")
            else:
                log.debug(f"Time spent on {self.context}: {time_spent:.4f} seconds")
            return result

        return wrapper  # type: ignore


class TrainingTimer:
    """Timer for timing the execution of code, aggregating over multiple training iterations.

    It is used as a context manager to measure the execution time of code and store the timing results
    for each function. The context managers can be nested.

    Attributes:
        results (dict): A dictionary to store timing results for various code.

    Example:
        timer = Timer()
        for i in range(100):
            with timer("func_a"):
                func_a()
        avg_time = sum(timer.results["func_a"]) / len(timer.results["func_a"])
        print(f"func_a() took {avg_time} seconds.")
    """

    def __init__(self) -> None:
        self.results = dict()
        self.average_results = dict()
        self.start_time = []
        self.func_stack = []
        self.reset()

    def reset(self) -> None:
        self.results = {key: [] for key in self.results}

    def __enter__(self) -> TrainingTimer:
        self.start_time.append(time.time())
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        end_time = time.time()
        result = end_time - self.start_time.pop()
        key = self.func_stack.pop()
        self.results.setdefault(key, [])
        self.results[key].append(result)

    def __call__(self, func_name: str) -> TrainingTimer:
        self.func_stack.append(func_name)
        return self

    def __getattr__(self, func_name: str) -> TrainingTimer:
        return self.__call__(func_name)

    def nested(self, func_name: str) -> TrainingTimer:
        return self.__call__(func_name)

    def compute_average_results(self) -> dict[str, float]:
        results = dict()
        for key, value_list in self.results.items():
            results[key] = sum(value_list) / len(value_list)
        return results


def timeout_handler(timeout_period: float, signum: int, frame: int) -> None:
    # What to do when the process gets stuck. For now, we simply end the process.
    error_message = f"Timeout error: more than {timeout_period} seconds passed since the last iteration."
    raise TimeoutError(error_message)


class Color:
    """A convenience class to colorize strings in the console.

    Example:
        import
        print("This is {Color.red('important')}.")
    """

    @staticmethod
    def red(x: str) -> str:
        return termcolor.colored(str(x), color="red")

    @staticmethod
    def green(x: str) -> str:
        return termcolor.colored(str(x), color="green")

    @staticmethod
    def cyan(x: str) -> str:
        return termcolor.colored(str(x), color="cyan")

    @staticmethod
    def yellow(x: str) -> str:
        return termcolor.colored(str(x), color="yellow")
