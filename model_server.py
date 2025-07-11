#!/usr/bin/env python3
"""
Main function that creates 8 worker processes with torchrun.
Workers have control loops to wait for input from the main function.
Worker processes are now handled by worker_sandbox.py.
"""

import argparse
import json
import os
import subprocess
import time
from typing import Dict, Any, Optional
from loguru import logger as log


def send_command_to_worker(rank: int, command: str, params: Optional[Dict[str, Any]] = None):
    """Send a command to a specific worker."""
    command_file = f"/tmp/worker_{rank}_commands.json"
    command_data = {"command": command, "params": params or {}}

    with open(command_file, "w") as f:
        json.dump(command_data, f)

    log.info(f"Sent command '{command}' to worker {rank}")


def get_worker_status(rank: int) -> Dict[str, Any]:
    """Get the status of a specific worker."""
    status_file = f"/tmp/worker_{rank}_status.json"

    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                return json.load(f)
        except Exception:
            return {"status": "unknown", "rank": rank}
    else:
        return {"status": "not_ready", "rank": rank}


def wait_for_workers_ready(num_workers: int, timeout: float = 30.0):
    """Wait for all workers to be ready."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        ready_count = 0
        for rank in range(num_workers):
            status = get_worker_status(rank)
            if status.get("status") == "ready":
                ready_count += 1

        if ready_count == num_workers:
            log.info(f"All {num_workers} workers are ready")
            return True

        time.sleep(0.5)

    log.error(f"Timeout waiting for workers to be ready. Only {ready_count}/{num_workers} ready")
    return False


def cleanup_worker_files(num_workers: int):
    """Clean up temporary worker files."""
    for rank in range(num_workers):
        for file_path in [f"/tmp/worker_{rank}_commands.json", f"/tmp/worker_{rank}_status.json"]:
            if os.path.exists(file_path):
                os.remove(file_path)


def main():
    """
    Main function that launches 8 worker processes using torchrun.
    """
    parser = argparse.ArgumentParser(description="Distributed Worker Manager")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker processes")
    parser.add_argument("--master_port", type=int, default=12345, help="Master port for distributed training")
    parser.add_argument("--backend", type=str, default="nccl", help="Backend for distributed training")

    args = parser.parse_args()

    # Clean up any existing worker files
    cleanup_worker_files(args.num_workers)

    log.info(f"Starting {args.num_workers} worker processes with torchrun")

    # Set up environment variables for distributed training
    env = os.environ.copy()
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = str(args.master_port)
    env["WORLD_SIZE"] = str(args.num_workers)

    # Build torchrun command to run worker_sandbox.py
    torchrun_cmd = [
        "torchrun",
        f"--nproc_per_node={args.num_workers}",
        f"--master_port={args.master_port}",
        "worker_sandbox.py",
    ]

    log.info(f"Running command: {' '.join(torchrun_cmd)}")

    # Launch worker processes
    try:
        process = subprocess.Popen(
            torchrun_cmd,
            env=env,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Give workers time to start
        time.sleep(2)

        # Wait for workers to be ready
        if not wait_for_workers_ready(args.num_workers):
            log.error("Failed to start all workers")
            process.terminate()
            return

        log.info("All workers started successfully!")

        # Example: Send tasks to all workers

        log.info("Sending example tasks to all workers...")

        # Send same task to all workers
        for rank in range(args.num_workers):
            send_command_to_worker(rank, "process_task", {"task_id": "example_task", "duration": 2.0})

        # Wait for tasks to complete
        log.info("Waiting for tasks to complete...")
        time.sleep(5)

        # Check results
        for rank in range(args.num_workers):
            status = get_worker_status(rank)
            log.info(f"Worker {rank} status: {status}")

        # Shutdown workers
        log.info("Shutting down workers...")
        for rank in range(args.num_workers):
            send_command_to_worker(rank, "shutdown")

        # Wait a bit for graceful shutdown
        time.sleep(2)

        # Terminate process if still running
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)

        log.info("All workers shut down")

    except Exception as e:
        log.error(f"Error: {e}")
    finally:
        # Clean up temporary files
        cleanup_worker_files(args.num_workers)


if __name__ == "__main__":
    main()
