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

    def __init__(self, num_workers: int):
        self.num_workers = num_workers

    def send_command_to_worker(self, rank: int, command: str, params: Optional[Dict[str, Any]] = None):
        command_file = f"/tmp/worker_{rank}_commands.json"
        command_data = {"command": command, "params": params or {}}

        with open(command_file, "w") as f:
            json.dump(command_data, f)

        log.info(f"Sent command '{command}' to worker {rank}")

    def cleanup_worker_files(self):
        for rank in range(self.num_workers):
            for file_path in [f"/tmp/worker_{rank}_commands.json", f"/tmp/worker_{rank}_status.json"]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    def send_task_to_all_workers(self, task_name: str, task_params: Dict[str, Any]):
        log.info(f"Sending task '{task_name}' to all workers...")

        for rank in range(self.num_workers):
            self.send_command_to_worker(rank, task_name, task_params)

    def shutdown_all_workers(self):
        log.info("Shutting down workers...")

        for rank in range(self.num_workers):
            self.send_command_to_worker(rank, "shutdown")


class WorkerStatus:

    def __init__(self, num_workers: int):
        self.num_workers = num_workers

    def _get_worker_status(self, rank: int, timeout: int = 1800) -> Dict[str, Any]:
        status_file = f"/tmp/worker_{rank}_status.json"
        start_time = time.time()

        while not os.path.exists(status_file):
            if time.time() - start_time > timeout:
                return {"status": "timeout", "rank": rank}
            time.sleep(0.5)

        try:
            with open(status_file, "r") as f:
                status = json.load(f)

            # remove status file so we can do a blocking wait for next status
            os.remove(status_file)

            assert os.path.exists(status_file) is False, "status file should be removed after processing"
            return status

        except Exception:
            log.error(f"Failed to read status file for worker {rank}")
            return {"status": "unknown", "rank": rank}

    """blocking call to wait for completion of all workers"""

    def wait_for_status(self, timeout: int = 1800) -> bool:
        statuses = {}

        # Collect statuses from all workers, ensure status file is removed after reading
        for rank in range(self.num_workers):
            statuses[rank] = self._get_worker_status(rank, timeout)

        for rank, worker_status in statuses.items():
            if worker_status.get("status") != "success":
                log.error(f"Worker {rank} failed: {worker_status}")
                return False
        log.info("All workers reported success")
        return True
