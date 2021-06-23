"""Functions to generate configurations"""
import os

from parsl import HighThroughputExecutor
from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.launchers import AprunLauncher, SimpleLauncher
from parsl.providers import LocalProvider
from parsl.monitoring import MonitoringHub


def local_config(log_dir: str, prefetch: int = 0) -> Config:
    """Single node with a single task per worker

    Args:
        log_dir: Path to store monitoring DB and parsl logs
        prefetch: Number of tasks for ML workers to prefetch for inference
    Returns:
        (Config) Parsl configuration
    """

    return Config(
        executors=[
            HighThroughputExecutor(
                address=address_by_hostname(),
                label="worker",
                max_workers=1,
                prefetch_capacity=prefetch,
                provider=LocalProvider(
                    nodes_per_block=1,
                    init_blocks=1,
                    max_blocks=1,
                    launcher=SimpleLauncher(),  # Places worker on the launch node
                ),
            )
        ],
        run_dir=log_dir,
        strategy='simple',
        max_idletime=15.
    )
