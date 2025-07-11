#!/usr/bin/env python3
"""
Worker sandbox for distributed processing.
Contains the worker processes that run in each distributed node.
"""

import json
import os
import sys
import time
import torch
import torch.distributed as dist
from loguru import logger as log


def worker_main():
    """
    Worker function that runs in each distributed process.
    Has a control loop to wait for input from the main function.
    """
    log.info(f"Worker init")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # Set device for this worker
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    log.info(f"Worker {rank}/{world_size} initialized on device {device}")

    # Create a simple control loop to wait for commands
    command_file = f"/tmp/worker_{rank}_commands.json"
    status_file = f"/tmp/worker_{rank}_status.json"

    # Signal that worker is ready
    with open(status_file, "w") as f:
        json.dump({"status": "ready", "rank": rank, "pid": os.getpid()}, f)

    try:
        while True:
            # Check for new commands
            if os.path.exists(command_file):
                try:
                    with open(command_file, "r") as f:
                        command_data = json.load(f)

                    command = command_data.get("command")
                    params = command_data.get("params", {})

                    log.info(f"Worker {rank} received command: {command}")

                    # Process different commands
                    if command == "shutdown":
                        log.info(f"Worker {rank} shutting down")
                        break
                    elif command == "process_task":
                        result = process_task(rank, params, device)

                        # Update status with result
                        with open(status_file, "w") as f:
                            json.dump(
                                {
                                    "status": "completed",
                                    "rank": rank,
                                    "task_id": params.get("task_id", "unknown"),
                                    "result": result,
                                },
                                f,
                            )
                    else:
                        log.warning(f"Worker {rank} received unknown command: {command}")
                        with open(status_file, "w") as f:
                            json.dump({"status": "error", "rank": rank, "error": f"Unknown command: {command}"}, f)

                    # Remove command file after processing
                    os.remove(command_file)

                except Exception as e:
                    log.error(f"Worker {rank} error processing command: {e}")
                    with open(status_file, "w") as f:
                        json.dump({"status": "error", "rank": rank, "error": str(e)}, f)
                    if os.path.exists(command_file):
                        os.remove(command_file)

            # Sleep briefly to avoid busy waiting
            time.sleep(0.1)

    except KeyboardInterrupt:
        log.info(f"Worker {rank} interrupted")
    finally:
        # Cleanup
        if os.path.exists(command_file):
            os.remove(command_file)
        if os.path.exists(status_file):
            os.remove(status_file)

        # Cleanup distributed
        if dist.is_initialized():
            dist.destroy_process_group()


def process_task(rank: int, params: dict, device: torch.device) -> float:

    task_id = params.get("task_id", "unknown")
    duration = params.get("duration", 1.0)

    log.info(f"Worker {rank} processing task {task_id} for {duration}s")

    # Simulate some work
    time.sleep(duration)

    # Create some dummy tensor computation
    x = torch.randn(100, 100, device=device)
    y = torch.matmul(x, x.T)
    result = torch.sum(y).item()

    log.info(f"Worker {rank} completed task {task_id}, result: {result:.4f}")

    return result


if __name__ == "__main__":

    worker_main()
