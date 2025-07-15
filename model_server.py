#!/usr/bin/env python3
"""
Main function that creates 8 worker processes with torchrun.
Workers have control loops to wait for input from the main function.
Worker processes are now handled by worker_sandbox.py.
"""

import argparse
import os
import subprocess
import time
from loguru import logger as log

from worker_manager import WorkerManager, WorkerStatus
from cosmos_transfer1.diffusion.inference.transfer import parse_arguments
import signal
import sys

args_worker = "--checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/example1_single_control_edge \
    --controlnet_specs assets/inference_cosmos_transfer1_single_control_edge.json \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus $NUM_GPU"


class ModelServer:
    """
    A class to manage distributed worker processes using torchrun.
    """

    def __init__(self, num_workers: int = 2, master_port: int = 12345, backend: str = "nccl"):

        self.num_workers = num_workers
        self.master_port = master_port
        # self.backend = backend
        self.process = None
        self.worker_manager = WorkerManager(num_workers)
        self.worker_status = WorkerStatus(num_workers)
        self._setup_environment()
        self.start_workers()

    def _setup_environment(self):
        self.env = os.environ.copy()
        self.env["MASTER_ADDR"] = "localhost"
        self.env["MASTER_PORT"] = str(self.master_port)
        self.env["WORLD_SIZE"] = str(self.num_workers)

    def start_workers(self):

        self.worker_manager.cleanup_worker_files()

        log.info(f"Starting {self.num_workers} worker processes with torchrun")

        torchrun_cmd = [
            "torchrun",
            f"--nproc_per_node={self.num_workers}",
            f"--master_port={self.master_port}",
            "worker_sandbox.py",  # TODO args_worker, for now we run worker with default args
        ]

        log.info(f"Running command: {' '.join(torchrun_cmd)}")

        # Launch worker processes
        try:
            self.process = subprocess.Popen(
                torchrun_cmd,
                env=self.env,
                # stdout=subprocess.PIPE,
                # stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            log.info("waiting for workers to start...")
            if not self.worker_status.wait_for_status():
                raise Exception("Failed to start all workers")

        except Exception as e:
            log.error(f"Error starting workers: {e}")
            self.stop_workers()
            raise e

    def stop_workers(self):
        if self.process is None:
            return

        self.worker_manager.shutdown_all_workers()

        # Wait a bit for graceful shutdown
        time.sleep(2)

        # Terminate process if still running
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)

        log.info("All workers shut down")
        self.process = None

    def run_inference(self, args: str):

        try:
            self.worker_manager.send_task_to_all_workers("process_task", {"duration": 2.0})

            # Wait for tasks to complete
            log.info("Waiting for tasks to complete...")
            if not self.worker_status.wait_for_status():
                log.info(f"inference failed")

        except Exception as e:
            log.error(f"Error during workflow: {e}")
        finally:
            self.stop_workers()
            self.worker_manager.cleanup_worker_files()

    def __del__(self):
        """Cleanup when the object is destroyed."""
        self.stop_workers()
        self.worker_manager.cleanup_worker_files()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        log.info("Exiting ModelServer context")
        self.stop_workers()
        self.worker_manager.cleanup_worker_files()


if __name__ == "__main__":

    def signal_handler(sig, frame):
        """Handle Ctrl+C and other termination signals."""
        log.info("Received interrupt signal, shutting down gracefully...")
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    num_gpus = int(os.environ.get("NUM_GPU", 1))
    with ModelServer(num_workers=num_gpus) as server:

        server.run_inference(args_worker)
