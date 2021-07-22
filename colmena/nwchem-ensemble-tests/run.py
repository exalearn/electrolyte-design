from typing import List, Union
from pathlib import Path
from datetime import datetime
from threading import Thread
from getpass import getuser
import argparse
import logging
import hashlib
import time
import json
import sys
import os

import pandas as pd
from colmena.method_server import ParslMethodServer
from colmena.models import Result
from colmena.redis.queue import ClientQueues, make_queue_pairs
from colmena.thinker import BaseThinker, result_processor, task_submitter
from colmena.thinker.resources import ResourceCounter
from psutil import Process, process_iter
from qcelemental.models import OptimizationResult, AtomicResult

from config import theta_nwchem_config
from moldesign.utils import get_platform_info


def track_memory_usage(out_path: str, write_frequency: float):
    """Track how busy the head node is

    Args:
        out_path: Path to the output file
        write_frequency: How often to write (s)
    """


    while True:
        # Get a timestamp
        ts = datetime.now().timestamp()
        
        # Measure the thinker process
        proc = Process()
        my_usage = proc.cpu_percent()
        my_memory = proc.memory_full_info().pss
        
        # Measure all processes from my user
        my_name = getuser()
        all_cpu = all_memory = 0
        for proc in process_iter():
            if proc.username() != my_name:
                continue
            try:
                all_cpu += proc.cpu_percent()
                all_memory += proc.memory_full_info().pss
            except:
                continue 
        
        with open(out_path, 'a') as fp:
            print(json.dumps({
                'time': ts,
                'thinker_cpu': my_usage,
                'thinker_mem': my_memory,
                'all_cpu': all_cpu,
                'all_mem': all_memory
            }), file=fp)
        time.sleep(write_frequency)


def run_simulation(smiles: str, n_nodes: int, mode: str) -> Union[OptimizationResult, AtomicResult]:
    """Run a single-point or relaxation computation computation

    Args:
        smiles: SMILES string to evaluate
        n_nodes: Number of nodes to use
        mode: What computation to perform: single, gradient, hessian, relax
    Returns:
        Result of the energy computation
    """
    from moldesign.simulate.functions import generate_inchi_and_xyz, relax_structure, run_single_point
    from moldesign.simulate.specs import get_qcinput_specification
    from qcelemental.models import DriverEnum
    
    # Make the initial geometry
    inchi, xyz = generate_inchi_and_xyz(smiles)

    # Make the compute spec
    compute_config = {'nnodes': n_nodes, 'cores_per_rank': 2}

    # Get the specification and make it more resilient
    spec, code = get_qcinput_specification('small_basis')
    if code == "nwchem":
        spec.keywords["dft__iterations"] = 150
        spec.keywords["geometry__noautoz"] = True

    # Compute the neutral geometry and hessian
    if mode == 'relax':
        _, _, neutral_relax = relax_structure(xyz, spec, compute_config=compute_config, charge=0, code=code)
        return neutral_relax
    else:
        return run_single_point(xyz, mode, spec, charge=0, compute_config=compute_config, code=code)


class Thinker(BaseThinker):
    """ML-enhanced optimization loop for molecular design"""

    def __init__(self, queues: ClientQueues,
                 output_dir: str,
                 task_queue: List[str],
                 qc_mode: str,
                 num_nodes: int,
                 nodes_per_qc: int):
        """
        Args:
            queues: Queues used to communicate with the method server
            output_dir: Path to write log files
            task_queue: List of molecules to evaluate as SMILES strings
            qc_mode: Type of QC computation to run
            num_nodes: Total number of nodes
        """
        super().__init__(queues, ResourceCounter(num_nodes, ['simulation']), daemon=True)

        # Configuration for the run
        self.nodes_per_qc = nodes_per_qc
        self.output_dir = Path(output_dir)
        self.task_queue = task_queue
        self.qc_mode = qc_mode

        # Allocate all resource to inference for the first task
        self.rec.reallocate(None, 'simulation', num_nodes)

    @task_submitter(task_type='simulation', n_slots=2)
    def submit_qc(self):
        # Wait until all slots free up
        if self.nodes_per_qc > 2:
            acq_success = self.rec.acquire('simulation', self.nodes_per_qc - 2, cancel_if=self.done)
            if not acq_success:
                raise ValueError('Node allocation failed')

        # Submit the next task
        smiles = self.task_queue.pop(0)
        self.logger.info(f'Submitted {smiles} to simulate with NWChem')
        self.queues.send_inputs(smiles, self.nodes_per_qc, self.qc_mode,
                                method='run_simulation', keep_inputs=True,
                                topic='simulate')

    @result_processor(topic='simulate')
    def process_outputs(self, result: Result):
        # Get basic task information
        smiles, n_nodes, _ = result.args
        
        # Release nodes for use by other processes
        self.rec.release("simulation", n_nodes)

        # If successful, add to the database
        if result.success:
            # Store the data in a molecule data object
            record = result.value

            # Write to disk
            with open(self.output_dir.joinpath('qcfractal-records.json'), 'a') as fp:
                print(record.json(), file=fp)
            self.logger.info(f'Added complete calculation for {smiles} to database.')
        else:
            self.logger.info(f'Computations failed for {smiles}. Check JSON file for stacktrace')

        # Write out the result to disk
        with open(self.output_dir.joinpath('simulation-results.json'), 'a') as fp:
            print(result.json(exclude={'value'}), file=fp)


if __name__ == '__main__':
    # User inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--redishost", default="127.0.0.1",
                        help="Address at which the redis server can be reached")
    parser.add_argument("--redisport", default="6379",
                        help="Port on which redis is available")
    parser.add_argument('--config', default='htex', choices=['htex', 'thread'], help='Name of the task server config')
    parser.add_argument('--search-space', help='Path to molecules to be screened', required=True)
    parser.add_argument('--max-evals', default=1024, help='Maximum number of molecules to evaluate', type=int)
    parser.add_argument('--nodes-per-task', help='Number of nodes per NWChem task.', default=2, type=int)
    parser.add_argument('--mode', default='hessian', help='What kind of QC computation to run')
    parser.add_argument('--random-seed', default=1, type=int, help='Random seed')

    # Parse the arguments
    args = parser.parse_args()
    run_params = args.__dict__

    # Define the compute setting for the system (only relevant for NWChem)
    nnodes = int(os.environ.get("COBALT_JOBSIZE", "1"))
    run_params["nnodes"] = nnodes
    run_params["qc_workers"] = nnodes / args.nodes_per_task

    # Create an output directory with the time and run parameters
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    out_dir = os.path.join('runs', f'N{nnodes}-n{args.nodes_per_task}-{args.mode}-{args.config}'
                                   f'-{start_time.strftime("%d%b%y-%H%M%S")}-{params_hash}')
    os.makedirs(out_dir, exist_ok=False)

    # Save the run parameters to disk
    with open(os.path.join(out_dir, 'run_params.json'), 'w') as fp:
        json.dump(run_params, fp, indent=2)
    with open(os.path.join(out_dir, 'environment.json'), 'w') as fp:
        json.dump(dict(os.environ), fp, indent=2)

    # Save the platform information to disk
    host_info = get_platform_info()
    with open(os.path.join(out_dir, 'host_info.json'), 'w') as fp:
        json.dump(host_info, fp, indent=2)

    # Set up the logging
    handlers = [logging.FileHandler(os.path.join(out_dir, 'runtime.log')),
                logging.StreamHandler(sys.stdout)]

    class ParslFilter(logging.Filter):
        """Filter out Parsl debug logs"""

        def filter(self, record):
            return not (record.levelno == logging.DEBUG and '/parsl/' in record.pathname)

    for h in handlers:
        h.addFilter(ParslFilter())

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, handlers=handlers)

    # Write the configuration
    # ML nodes: N for updating models, 1 for MolDQN, 1 for inference runs
    config = theta_nwchem_config(args.config, os.path.join(out_dir, 'run-info'),
                                 nodes_per_nwchem=args.nodes_per_task, total_nodes=nnodes)

    # Save Parsl configuration
    with open(os.path.join(out_dir, 'parsl_config.txt'), 'w') as fp:
        print(str(config), file=fp)

    # Load in the search space
    search_space = pd.read_csv(args.search_space, delim_whitespace=True)
    search_space = search_space.sample(args.max_evals, random_state=args.random_seed)
    logging.info(f'Read {len(search_space)} from {args.search_space} and shuffled them.'
                 f' First: {search_space["smiles"].iloc[:5].values}')

    # Connect to the redis server
    client_queues, server_queues = make_queue_pairs(args.redishost, args.redisport,
                                                    serialization_method="pickle",
                                                    topics=['simulate', 'infer', 'train'],
                                                    keep_inputs=False)

    # Create the method server and task generator
    doer = ParslMethodServer([run_simulation], server_queues, config)

    # Configure the "thinker" application
    thinker = Thinker(client_queues,
                      out_dir,
                      search_space['smiles'].tolist(),
                      args.mode,
                      nnodes,
                      args.nodes_per_task)
    logging.info('Created the method server and task generator')

    # Start the usage tracker
    thr = Thread(target=track_memory_usage, args=(os.path.join(out_dir, 'usage.json'), 15), daemon=True)
    thr.start()

    try:
        # Launch the servers
        #  The method server is a Thread, so that it can access the Parsl DFK
        #  The task generator is a Thread, so that all debugging methods get cast to screen
        doer.start()
        thinker.start()
        logging.info(f'Running on {os.getpid()}')
        logging.info('Launched the servers')

        # Wait for the task generator to complete
        thinker.join()
        logging.info('Task generator has completed')
    finally:
        client_queues.send_kill_signal()

    # Wait for the method server to complete
    doer.join()
