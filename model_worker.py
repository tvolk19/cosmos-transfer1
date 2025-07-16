#!/usr/bin/env python3
"""
Worker sandbox for distributed processing.
Contains the worker processes that run in each distributed node.
"""

import json
import os
import sys
import time
import traceback
import torch.distributed as dist
from loguru import logger as log
from cosmos_transfer1.diffusion.inference.transfer_pipeline import TransferPipeline
import sys
from loguru import logger
from typing import Dict, Any, Optional

# Configure loguru with custom color for this worker process
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <white>{level}</white> | <green>{name}</green>:<yellow>{function}</yellow>:<yellow>{line}</yellow> - <white>{message}</white>",
    level="INFO",
)

"""
this function is running in processes started by torchrun.
# TODO: dynamically load model pipeline based on MODEL_MODULE and MODEL_CLASS
# TODO: pass through env var for CHECKPOINT_DIR, so far we use default
"""


def create_status(status_file: str, rank: int, status: str, message: str = "") -> None:
    with open(status_file, "w") as f:
        json.dump(
            {"rank": rank, "status": status, "result": message},
            f,
        )


def worker_main():
    """
    Worker function that runs in each distributed process.
    Has a control loop to wait for input from the main function.
    """
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    log.info(f"Worker init {rank}/{world_size}")

    # Create a simple control loop to wait for commands
    command_file = f"/tmp/worker_{rank}_commands.json"
    status_file = f"/tmp/worker_{rank}_status.json"

    try:

        pipeline = None
        if os.getenv("MODEL_MODULE") and os.getenv("MODEL_CLASS"):
            # Dynamically import the model module and class
            model_module = os.getenv("MODEL_MODULE")
            model_class = os.getenv("MODEL_CLASS")
            # module = __import__(model_module, fromlist=[model_class])
            # model_class = getattr(module, model_class)
            log.info(f"Using model {model_class} from module {model_module}")
            pipeline = TransferPipeline(num_gpus=world_size)
        else:
            log.error("MODEL_MODULE and MODEL_CLASS environment variables are not set.")

        # Signal that worker is ready
        with open(status_file, "w") as f:
            json.dump({"status": "success", "rank": rank, "pid": os.getpid()}, f)

        while True:
            # Check for new commands
            if os.path.exists(command_file):
                try:
                    with open(command_file, "r") as f:
                        command_data = json.load(f)

                    command = command_data.get("command")
                    params = command_data.get("params", {})

                    log.info(f"Worker {rank} received command: {command}")
                    log.info(f"Worker {rank} running inference with parameters: {params}")

                    # Process different commands
                    if command == "process_task":

                        # read input parameters from command
                        if pipeline:
                            # Create model parameters from params
                            params = TransferPipeline.create_model_params(**params)
                            pipeline.infer(params)

                        # Update status with result
                        create_status(status_file, rank, "success", "result_placeholder")
                    elif command == "shutdown":
                        log.info(f"Worker {rank} shutting down")
                        break
                    else:
                        log.warning(f"Worker {rank} received unknown command: {command}")
                        create_status(status_file, rank, "error", f"Unknown command: {command}")

                    # Remove command file after processing
                    os.remove(command_file)

                except Exception as e:
                    log.error(f"Worker {rank} error processing command: {e}")
                    error_trace = traceback.format_exc()
                    log.error(f"Stack trace:\n{error_trace}")

                    create_status(status_file, rank, "error", str(e))
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


if __name__ == "__main__":

    worker_main()
