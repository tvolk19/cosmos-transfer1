#!/usr/bin/env python3
"""
WorkerManager class for managing communication with distributed worker processes.

"""

import json
import os
import time
from typing import Dict, Any, Optional
from loguru import logger as log


class WorkerCommand:

    def __init__(self, num_workers: int):
        self.num_workers = num_workers

    def _send_command_to_worker(self, rank: int, command: str, params: Optional[Dict[str, Any]] = None):
        command_file = f"/tmp/worker_{rank}_commands.json"
        command_data = {"command": command, "params": params or {}}

        with open(command_file, "w") as f:
            json.dump(command_data, f)

        log.info(f"Sent command '{command}' to worker {rank}")

    def cleanup(self):
        for rank in range(self.num_workers):
            for file_path in [f"/tmp/worker_{rank}_commands.json"]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    # asynchronous command to all workers
    def broadcast(self, task_name: str, task_params: Dict[str, Any]):
        log.info(f"Broadcasting task '{task_name}' to all workers...")

        for rank in range(self.num_workers):
            self._send_command_to_worker(rank, task_name, task_params)

    def wait_for_command(self, rank: int) -> Optional[Dict[str, Any]]:
        command_file = f"/tmp/worker_{rank}_commands.json"
        log.info(f"worker {rank}: Waiting for command file {command_file}")
        while not os.path.exists(command_file):
            time.sleep(0.5)

        try:
            with open(command_file, "r") as f:
                command_data = json.load(f)
            os.remove(command_file)  # Remove command file after reading
            return command_data
        except Exception as e:
            log.error(f"Failed to read command file for worker {rank}: {e}")
            raise e


class WorkerStatus:

    def __init__(self, num_workers: int):
        self.num_workers = num_workers

    def signal_status(self, rank: int, status: str, message: str = "") -> None:
        status_file = f"/tmp/worker_{rank}_status.json"

        log.info(f"worker {rank} status: {status}, message: {message}")
        with open(status_file, "w") as f:
            json.dump(
                {"rank": rank, "status": status, "result": message},
                f,
            )

    def _get_worker_status(self, rank: int, timeout: int = 1800) -> Dict[str, Any]:
        status_file = f"/tmp/worker_{rank}_status.json"
        start_time = time.time()

        while not os.path.exists(status_file):
            if time.time() - start_time > timeout:
                os.remove(status_file)
                return {"status": "timeout", "rank": rank}
            time.sleep(0.5)

        try:
            with open(status_file, "r") as f:
                status = json.load(f)

            # remove status file so we can do a blocking wait for next status
            log.info(f"Worker {rank} removing status file {status_file}")
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
            # return on first error
            # TODO we need to report the most severe error
            # in case of timeout we most likely have a hang. then we should restart the worker?
            if worker_status.get("status") != "success":
                log.error(f"Worker {rank} failed: {worker_status}")
                return False
        log.info("All workers reported success")
        return True

    def cleanup(self):
        for rank in range(self.num_workers):
            for file_path in [f"/tmp/worker_{rank}_status.json"]:
                if os.path.exists(file_path):
                    os.remove(file_path)
