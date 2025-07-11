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

from worker_manager import WorkerManager
from cosmos_transfer1.diffusion.inference.transfer import parse_arguments

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
        """
        Initialize the ModelServer.

        Args:
            num_workers: Number of worker processes to spawn
            master_port: Port for distributed training coordination
            backend: Backend for distributed training (nccl, gloo, etc.)
        """
        self.num_workers = num_workers
        self.master_port = master_port
        self.backend = backend
        self.process = None
        self.worker_manager = WorkerManager(num_workers)
        self._setup_environment()
        self.start_workers()

    def _setup_environment(self):
        """Set up environment variables for distributed training."""
        self.env = os.environ.copy()
        self.env["MASTER_ADDR"] = "localhost"
        self.env["MASTER_PORT"] = str(self.master_port)
        self.env["WORLD_SIZE"] = str(self.num_workers)

    def start_workers(self):
        # Clean up any existing worker files
        self.worker_manager.cleanup_worker_files()

        log.info(f"Starting {self.num_workers} worker processes with torchrun")

        # Build torchrun command to run worker_sandbox.py
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

            # Give workers time to start
            time.sleep(2)

            # Wait for workers to be ready
            if not self.worker_manager.wait_for_workers_ready():
                log.error("Failed to start all workers")
                self.stop_workers()
                raise Exception("Failed to start all workers")

            log.info("All workers started successfully!")

        except Exception as e:
            log.error(f"Error starting workers: {e}")
            raise e

    def stop_workers(self):
        """Shutdown all worker processes."""
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
            self.worker_manager.send_task_to_all_workers("process_task", {"task_id": "example_task", "duration": 2.0})

            # Wait for tasks to complete
            log.info("Waiting for tasks to complete...")
            time.sleep(5)

            # todo this should be blocking call to wait for completion of all workers
            statuses = self.worker_manager.get_all_worker_statuses()
            for rank, status in statuses.items():
                log.info(f"Worker {rank} status: {status}")

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
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_workers()
        self.worker_manager.cleanup_worker_files()


def main():
    """
    Main function that launches worker processes using the ModelServer class.
    """
    parser = argparse.ArgumentParser(description="Distributed Worker Manager")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker processes")
    parser.add_argument("--master_port", type=int, default=12345, help="Master port for distributed training")
    parser.add_argument("--backend", type=str, default="nccl", help="Backend for distributed training")

    args = parser.parse_args()

    # Create and run the model server
    with ModelServer(num_workers=args.num_workers, master_port=args.master_port, backend=args.backend) as server:

        # todo should we pass in a argparse.Namespace object?
        # args = parse_arguments(args)

        server.run_inference(args_worker)

    # todo this should be blocking call


if __name__ == "__main__":
    main()
