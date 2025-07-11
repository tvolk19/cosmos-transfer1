#!/usr/bin/env python3
"""
WorkerManager class for managing communication with distributed worker processes.
"""

import json
import os
import time
from typing import Dict, Any, Optional
from loguru import logger as log


class WorkerManager:
    """
    A class to manage communication with distributed worker processes.
    """

    def __init__(self, num_workers: int):
        """
        Initialize the WorkerManager.

        Args:
            num_workers: Number of worker processes to manage
        """
        self.num_workers = num_workers

    def send_command_to_worker(self, rank: int, command: str, params: Optional[Dict[str, Any]] = None):
        """Send a command to a specific worker."""
        command_file = f"/tmp/worker_{rank}_commands.json"
        command_data = {"command": command, "params": params or {}}

        with open(command_file, "w") as f:
            json.dump(command_data, f)

        log.info(f"Sent command '{command}' to worker {rank}")

    def get_worker_status(self, rank: int) -> Dict[str, Any]:
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

    def wait_for_workers_ready(self, timeout: float = 30.0) -> bool:
        """Wait for all workers to be ready."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            ready_count = 0
            for rank in range(self.num_workers):
                status = self.get_worker_status(rank)
                if status.get("status") == "ready":
                    ready_count += 1

            if ready_count == self.num_workers:
                log.info(f"All {self.num_workers} workers are ready")
                return True

            time.sleep(0.5)

        log.error(f"Timeout waiting for workers to be ready. Only {ready_count}/{self.num_workers} ready")
        return False

    def cleanup_worker_files(self):
        """Clean up temporary worker files."""
        for rank in range(self.num_workers):
            for file_path in [f"/tmp/worker_{rank}_commands.json", f"/tmp/worker_{rank}_status.json"]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    def send_task_to_all_workers(self, task_name: str, task_params: Dict[str, Any]):
        """Send a task to all workers."""
        log.info(f"Sending task '{task_name}' to all workers...")

        for rank in range(self.num_workers):
            self.send_command_to_worker(rank, task_name, task_params)

    def get_all_worker_statuses(self) -> Dict[int, Dict[str, Any]]:
        """Get status of all workers."""
        statuses = {}
        for rank in range(self.num_workers):
            statuses[rank] = self.get_worker_status(rank)
        return statuses

    def shutdown_all_workers(self):
        """Send shutdown command to all workers."""
        log.info("Shutting down workers...")

        for rank in range(self.num_workers):
            self.send_command_to_worker(rank, "shutdown")
