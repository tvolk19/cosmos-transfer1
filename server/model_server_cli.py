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
import os
import subprocess
import json
from cosmos_transfer1.utils import log


class ModelServerCli:
    """Most basic server using an existing CLI model.

    This server has no communication channel with the workers, so no errors are reported."""

    def __init__(self, num_workers: int = 8, checkpoint_dir: str = "checkpoints"):
        """Initialize the model server and start worker processes.
        Args:
            num_workers (int): Number of worker processes to create (default: 2)

        """

        self.num_workers = num_workers
        self.checkpoint_dir = checkpoint_dir
        self.process = None
        self._setup_environment()

    def _setup_environment(self):
        self.env = os.environ.copy()

    def infer(self, args: dict, output_dir: str = "outputs/"):
        """Execute inference across all worker processes."""

        log.info(f"Starting {self.num_workers} worker processes with torchrun")
        cli_cmd = "cosmos_transfer1/diffusion/inference/transfer.py"

        log.debug(json.dumps(args, indent=2))
        # Save arguments to JSON file in output_dir
        os.makedirs(output_dir, exist_ok=True)
        args_file = os.path.join(output_dir, "inference_args.json")
        with open(args_file, "w") as f:
            json.dump(args, f, indent=2)
        log.info(f"Saved inference arguments to {args_file}")

        torchrun_cmd = [
            "torchrun",
            f"--nproc_per_node={self.num_workers}",
            "--nnodes=1",
            "--node_rank=0",
            cli_cmd,
            f"--checkpoint_dir={self.checkpoint_dir}",
            f"--num_gpus={self.num_workers}",
            f"--controlnet_specs={args_file}",
            f"--video_save_folder={output_dir}",
        ]

        log.info(f"Running command: {' '.join(torchrun_cmd)}")

        # Launch worker processes
        try:
            self.process = subprocess.Popen(
                torchrun_cmd,
                env=self.env,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Wait for the process to complete
            return_code = self.process.wait()

            if return_code == 0:
                log.info("Inference completed successfully")
            else:
                log.error(f"Inference failed with return code: {return_code}")
                raise subprocess.CalledProcessError(return_code, torchrun_cmd)

        except Exception as e:
            log.error(f"Error running inference: {e}")
            raise e

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection."""
        self.cleanup()

    def __enter__(self):
        """Enter the context manager.

        Returns:
            ModelServer: Self reference for context manager usage
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and perform cleanup.
        Ensures proper cleanup regardless of whether an exception occurred.
        """
        log.info("Exiting ModelServer context")
        self.cleanup()

    def cleanup(self):
        """Clean up server resources and shutdown workers."""
        log.info("Cleaning up ModelServer")
