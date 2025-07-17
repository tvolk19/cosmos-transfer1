#!/usr/bin/env python3
"""
Main function that creates 8 worker processes with torchrun.
Workers have control loops to wait for input from the main function.
Worker processes are now handled by worker_sandbox.py.
"""

import os
import subprocess
import time
from loguru import logger as log
import json
import signal
import sys
from worker_manager import WorkerCommand, WorkerStatus
from cosmos_transfer1.diffusion.inference.transfer_pipeline import TransferPipeline


class ModelServer:
    """
    A class to manage distributed worker processes using torchrun.
    """

    def __init__(self, num_workers: int = 2, master_port: int = 12345, backend: str = "nccl"):

        self.num_workers = num_workers
        self.master_port = master_port
        # self.backend = backend
        self.process = None
        self.worker_command = WorkerCommand(num_workers)
        self.worker_status = WorkerStatus(num_workers)
        self._setup_environment()
        self.start_workers()

    def _setup_environment(self):
        self.env = os.environ.copy()
        self.env["MASTER_ADDR"] = "localhost"
        self.env["MASTER_PORT"] = str(self.master_port)
        self.env["WORLD_SIZE"] = str(self.num_workers)

    def start_workers(self):

        self.worker_command.cleanup()
        self.worker_status.cleanup()

        log.info(f"Starting {self.num_workers} worker processes with torchrun")

        torchrun_cmd = [
            "torchrun",
            f"--nproc_per_node={self.num_workers}",
            f"--master_port={self.master_port}",
            "model_worker.py",  # TODO args_worker, for now we run worker with default args
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

        self.worker_command.broadcast("shutdown", {})

        # Wait a bit for graceful shutdown
        time.sleep(2)

        # Terminate process if still running
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)

        log.info("All workers shut down")
        self.process = None

    def send_request(self, args: dict):

        try:
            self.worker_command.broadcast("inference", args)

            log.info("Waiting for tasks to complete...")
            if not self.worker_status.wait_for_status():
                log.error(f"inference failed for some workers")

        except Exception as e:
            log.error(f"Error during workflow: {e}")
        finally:
            # todo should the worker consume command and clean up?
            self.worker_command.cleanup()

    def __del__(self):
        self.cleanup

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        log.info("Exiting ModelServer context")
        self.cleanup()

    def cleanup(self):
        log.info("Cleaning up ModelServer")
        self.stop_workers()
        self.worker_command.cleanup()
        self.worker_status.cleanup()


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

        args, args_dict = TransferPipeline.validate_params()

        server.send_request(args_dict)
        server.send_request(args_dict)
