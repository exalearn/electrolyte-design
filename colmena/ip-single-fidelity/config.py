"""Functions to generate configurations"""
import os

from parsl import HighThroughputExecutor
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
        ml_prefetch: Number of tasks for ML workers to prefect
    Returns:
        (Config) Parsl configuration
    """
    assert total_nodes % nodes_per_nwchem == 0, "NWChem node count not a multiple of nodes per task"
    nwc_workers = total_nodes // nodes_per_nwchem

    return Config(
        executors=[
            HighThroughputExecutor(
                address=address_by_hostname(),
                label="qc",
                max_workers=nwc_workers,
                cores_per_worker=1e-6,
                provider=LocalProvider(
                    nodes_per_block=1,
                    init_blocks=0,
                    max_blocks=1,
                    launcher=SimpleLauncher(),  # Places worker on the launch node
                    worker_init='''
module load miniconda-3
conda activate /lus/theta-fs0/projects/CSC249ADCD08/edw/env
''',
                ),
            ),
            HighThroughputExecutor(
                address=address_by_hostname(),
                label="ml",
                max_workers=1,
                prefetch_capacity=ml_prefetch,
                provider=LocalProvider(
                    nodes_per_block=nodes_per_nwchem,
                    init_blocks=0,
                    max_blocks=total_nodes // nodes_per_nwchem,  # Limits the number of manager processes,
                    launcher=AprunLauncher(overrides='-d 64 --cc depth'),  # Places worker on the compute node
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
