"""Functions to generate configurations"""
import os

from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.launchers import AprunLauncher, SimpleLauncher
from parsl.providers import LocalProvider
from parsl.monitoring import MonitoringHub


def theta_nwchem_config(
        choice: str,
        log_dir: str,
        nodes_per_nwchem: int = 2,
        total_nodes: int = int(os.environ.get("COBALT_JOBSIZE", 1))) -> Config:
    """Theta configuration to run NWChem

    Args:
        choice: Choice of the runtime configuration
        nodes_per_nwchem: Number of nodes per NWChem computation
        log_dir: Path to store monitoring DB and parsl logs
        total_nodes: Total number of nodes available. Default: COBALT_JOBSIZE
    Returns:
        (Config) Parsl configuration
    """
    assert total_nodes % nodes_per_nwchem == 0, "NWChem node count not a multiple of nodes per task"
    nwc_workers = total_nodes // nodes_per_nwchem

    if choice == "htex":
        qc_exec = HighThroughputExecutor(
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
        )
    elif choice=='thread':
        qc_exec = ThreadPoolExecutor(
            label='qc',
            max_threads=nwc_workers
        )
    else:
        raise ValueError(f'Choice "{choice}" not recognized ')

    return Config(
        executors=[qc_exec],
        run_dir=log_dir,
        strategy='simple',
        max_idletime=15.
    )
