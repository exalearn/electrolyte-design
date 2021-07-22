"""Functions to generate configurations"""
import os

from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.launchers import AprunLauncher, SimpleLauncher
from parsl.providers import LocalProvider
from parsl.monitoring import MonitoringHub


def theta_nwchem_config(log_dir: str,
                        nodes_per_nwchem: int = 2,
                        total_nodes: int = int(os.environ.get("COBALT_JOBSIZE", 1)),
                        ml_prefetch: int = 0) -> Config:
    """Theta configuration where QC workers sit on the launch node (to be able to aprun)
    and ML workers are placed on compute nodes

    Args:
        nodes_per_nwchem: Number of nodes per NWChem computation
        log_dir: Path to store monitoring DB and parsl logs
        total_nodes: Total number of nodes available. Default: COBALT_JOBSIZE
        ml_prefetch: Number of tasks for ML workers to prefetch for inference
    Returns:
        (Config) Parsl configuration
    """
    assert total_nodes % nodes_per_nwchem == 0, "NWChem node count not a multiple of nodes per task"
    nwc_workers = total_nodes // nodes_per_nwchem

    return Config(
        executors=[
            ThreadPoolExecutor(
                label='qc',
                max_threads=nwc_workers
            ),
            HighThroughputExecutor(
                address=address_by_hostname(),
                label="ml-inference",
                max_workers=1,
                prefetch_capacity=ml_prefetch,
                provider=LocalProvider(
                    nodes_per_block=nodes_per_nwchem,
                    init_blocks=0,
                    max_blocks=total_nodes // nodes_per_nwchem,  # Limits the number of manager processes,
                    launcher=AprunLauncher(overrides='-d 256 --cc depth -j 4'),  # Places worker on the compute node
                    worker_init='''
module load miniconda-3
conda activate /lus/theta-fs0/projects/CSC249ADCD08/edw/env
    ''',
                ),
            ),
            HighThroughputExecutor(
                address=address_by_hostname(),
                label="ml-train",
                max_workers=1,
                prefetch_capacity=0,
                provider=LocalProvider(
                    nodes_per_block=nodes_per_nwchem,
                    init_blocks=0,
                    max_blocks=nwc_workers,  # Limits the number of manager processes,
                    launcher=AprunLauncher(overrides='-d 256 --cc depth -j 4'),  # Places worker on the compute node
                    worker_init='''
module load miniconda-3
conda activate /lus/theta-fs0/projects/CSC249ADCD08/edw/env
    ''',
                ),
            )
        ],
        monitoring=MonitoringHub(
            hub_address=address_by_hostname(),
            monitoring_debug=False,
            resource_monitoring_interval=10,
            logdir=log_dir,
            logging_endpoint=f'sqlite:///{os.path.join(log_dir, "monitoring.db")}'
        ),
        run_dir=log_dir,
        strategy='simple',
        max_idletime=15.
    )
