"""Functions to generate configurations"""
import os

from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.launchers import AprunLauncher, SimpleLauncher, SingleNodeLauncher
from parsl.providers import LocalProvider, CobaltProvider
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
                label="ml",
                max_workers=1,
                prefetch_capacity=ml_prefetch,
                provider=LocalProvider(
                    nodes_per_block=nodes_per_nwchem,  # Minimum increment in blcoks
                    init_blocks=0,
                    max_blocks=total_nodes // nodes_per_nwchem,  # Limits the number of manager processes,
                    launcher=AprunLauncher(overrides='-d 256 --cc depth -j 4'),  # Places worker on the compute node
                    worker_init='''
module load miniconda-3
conda activate /lus/theta-fs0/projects/CSC249ADCD08/edw/env
    ''',
                ),
            )],
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


def theta_persistent(log_dir: str,
                     nodes_per_nwchem: int = 1,
                     qc_nodes: int = 8,
                     ml_nodes: int = 8,
                     ml_prefetch: int = 0) -> Config:
    """Configuration where the application is persistent and sits on the Theta login node.

    Nodes will be requested from Cobalt using separate jobs for ML and QC tasks.

    Args:
        nodes_per_nwchem: Number of nodes per NWChem computation
        log_dir: Path to store monitoring DB and parsl logs
        qc_nodes: Number of nodes dedicated to QC tasks
        ml_prefetch: Number of tasks for ML workers to prefetch for inference
    Returns:
        (Config) Parsl configuration
    """
    return Config(
        retries=8,
        executors=[
            HighThroughputExecutor(
                address=address_by_hostname(),
                label="qc",
                max_workers=qc_nodes // nodes_per_nwchem,
                prefetch_capacity=ml_prefetch,
                provider=CobaltProvider(
                    account='CSC249ADCD08',
                    queue='debug-cache-quad',
                    walltime='00:60:00',
                    nodes_per_block=qc_nodes,
                    init_blocks=0,
                    max_blocks=1,
                    launcher=SingleNodeLauncher(),
                    worker_init='''
module load miniconda-3
conda activate /lus/theta-fs0/projects/CSC249ADCD08/edw/env


export OMP_NUM_THREADS=64
export KMP_INIT_AT_FORK=FALSE

export PATH="/lus/theta-fs0/projects/CSC249ADCD08/software/nwchem-6.8.1/bin/LINUX64:$PATH"
mkdir -p scratch  # For the NWChem tasks
pwd
which nwchem
hostname
module load atp
export MPICH_GNI_MAX_EAGER_MSG_SIZE=16384
export MPICH_GNI_MAX_VSHORT_MSG_SIZE=10000
export MPICH_GNI_MAX_EAGER_MSG_SIZE=131072
export MPICH_GNI_NUM_BUFS=300
export MPICH_GNI_NDREG_MAXSIZE=16777216
export MPICH_GNI_MBOX_PLACEMENT=nic
export MPICH_GNI_LMT_PATH=disabled
export COMEX_MAX_NB_OUTSTANDING=6
export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64_lin/:/opt/intel/compilers_and_libraries_2020.0.166/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH
''',
                ),
            ),
            HighThroughputExecutor(
                address=address_by_hostname(),
                label="ml",
                max_workers=1,
                prefetch_capacity=ml_prefetch,
                provider=CobaltProvider(
                    account='CSC249ADCD08',
                    queue='debug-flat-quad',
                    nodes_per_block=ml_nodes,
                    scheduler_options='#COBALT --attrs enable_ssh=1',
                    walltime='00:60:00',
                    init_blocks=0,
                    max_blocks=1,
                    launcher=AprunLauncher(overrides='-d 256 --cc depth -j 4'),  # Places worker on the compute node
                    worker_init='''
module load miniconda-3
conda activate /lus/theta-fs0/projects/CSC249ADCD08/edw/env''',
                ),
            )],
        run_dir=log_dir,
        strategy='simple',
        max_idletime=15.
    )
