import pickle as pkl
import argparse
import hashlib
import json
import sys
import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import sample, choice, shuffle, random
from datetime import datetime
from functools import partial, update_wrapper
from queue import Queue, Empty
from threading import Event, Lock, Thread, Semaphore
from typing import List, Dict, Tuple, Optional
from traceback import TracebackException

import numpy as np
from pydantic import BaseModel
from moldesign.score.mpnn import evaluate_mpnn
from moldesign.simulate.functions import compute_atomization_energy
from moldesign.simulate.specs import lookup_reference_energies, get_qcinput_specification
from moldesign.utils import get_platform_info

from colmena.thinker import BaseThinker, agent
from colmena.method_server import ParslMethodServer
from colmena.redis.queue import ClientQueues, make_queue_pairs
from config import theta_nwchem_config, theta_xtb_config


class Thinker(BaseThinker):
    """Screen-then-rank-then-run molecular design loop"""

    def __init__(self, queues: ClientQueues,
                 search_space_path: str,
                 mpnn_paths: List[str],
                 output_dir: str,
                 n_parallel_qc: int,
                 n_parallel_ml: int,
                 molecules_per_ml_task: int,
                 queue_length: int,
                 sampling_fraction: Optional[float],
                 excess_ml_tasks: int):
        """
        Args:
            queues (ClientQueues): Queues to use to communicate with server
            search_path_path: Path to molecule search space
            mpnn_paths: Paths to an ensemble of pre-trained MPNNs
            output_dir (str): Path to the run directory
            molecules_per_ml_task (int): Number of molecules to send with each ML task
            n_parallel_qc (int): Maximum number of QC calculations to perform in parallel
            n_parallel_ml (int): Maximum number of ML calculations to perform in parallel
            queue_length (int): Number of molecules to rank
            sampling_fraction (float): Fraction of search space to explore
            excess_ml_tasks (int): Number of excess tasks to keep in queue above the 
                number necessary to keep the ML workers
        """
        super().__init__(queues, daemon=True)

        # Generic stuff: logging, communication to Method Server
        self.output_dir = output_dir
        
        # The ML components
        self.mpnn_paths = mpnn_paths
        self.search_space_path = search_space_path
        self.sampling_fraction = sampling_fraction
        
        # Attributes associated with the parallelism/problem size
        self.n_parallel_qc = n_parallel_qc
        self.molecules_per_ml_task = molecules_per_ml_task
        
        # Compute the number of concurrent ML tasks
        ml_queue_length = n_parallel_ml + excess_ml_tasks

        # Synchronization between the threads
        self.queue_length = queue_length
        self._task_queue = Queue(maxsize=queue_length)
        self._inference_queue = Queue(maxsize=n_parallel_ml)
        self._ml_task_pool = Semaphore(ml_queue_length)
        self._qc_task_pool = Semaphore(self.n_parallel_qc)
        self._ml_tasks_submitted = Event()
        self.n_ml_tasks = 0  # Number active

    def _write_result(self, result: BaseModel, filename: str, keep_inputs: bool = True, keep_outputs: bool = True):
        """Write result to a log file

        Args:
            result: Result to be written
            filename: Name of the log file
            keep_inputs: Whether to write the function inputs
            keep_outputs: Whether to write the function outputs
        """

        # Determine which fields to dump
        exclude = set()
        if not keep_inputs:
            exclude.add('inputs')
        if not keep_outputs:
            exclude.add('value')

        # Write it out
        with open(os.path.join(self.output_dir, filename), 'a') as fp:
            print(result.json(exclude=exclude), file=fp)

    @agent
    def simulation_consumer(self):
        """Submit and process simulation tasks"""
      
        # As they come back submit new ones
        for i in range(self.queue_length):
            # Get the task and store its content
            result = self.queues.get_result(topic='simulate')
            self._qc_task_pool.release()
            self.logger.info(f'Retrieved completed QC task {i+1}/{self.queue_length}')

            # Store the content from the previous run
            if result.success:               
                # Save the data
                self._write_result(result.value[1], 'qcfractal_records.jsonld')
                if result.value[2] is not None:
                    self._write_result(result.value[2], 'qcfractal_records.jsonld')
                result.value = result.value[0]  # Do not store the full results in the database
            else:
                self.logger.warning('Calculation failed! See simulation outputs and Parsl log file')
            self._write_result(result, 'simulation_records.jsonld', keep_outputs=True)
            
            
    @agent
    def search_space_reader(self):
        """Reads search space from disk. 
        
        Separate thread to keep a queue of molecules ready to submit"""
        
        with open(self.search_space_path) as fp:
            self.logger.info(f'Opened search space molecules from: {self.search_space_path}')
            
            # Compute the number of entries to pull to get desired sampling rate
            if self.sampling_fraction is None or self.sampling_fraction >= 1.0:
                chunk_size = self.molecules_per_ml_task
            else:
                chunk_size = int(self.molecules_per_ml_task / self.sampling_fraction)
            
            # Loop until out of molecules
            is_done = False
            while not is_done:
                # Create a chunk
                chunk = [line for line, _ in zip(fp, range(chunk_size))]
                is_done = len(chunk) != chunk_size  # Done if we do not reach the desired chunk size
                if is_done:
                    self.logger.info('Pulled the last batch of molecules')
                    
                # Downsample the chunk to the desired size
                if self.sampling_fraction is not None:
                    desired_size = int(len(chunk) * self.sampling_fraction)
                    self.logger.info(f'Downsampling batch from {len(chunk)} to {desired_size}')
                    chunk = sample(chunk, desired_size)                    

                # Parse out the SMILES strings
                chunk = [line.strip().split(",")[-1] for line in chunk]  # Molecule is the last entry in line
                
                # Put it in the queue for the task submitter thread
                self._inference_queue.put(chunk)
                
            # Put a flag in the queue to say we are done
            self._inference_queue.put(None)

            
    @agent
    def ml_task_submitter(self):
        self.n_ml_tasks = 0 
       
        # Submit all of the ML tasks
        while True:
            # Get a chunk that is ready to submit
            chunk = self._inference_queue.get()
            if chunk is None:
                self.logger.info('No more inference tasks to submit')
                break
            
            # Acquire permission to submit to the queue 
            #  We do not want too many tasks to be submitted at once to control memory usage
            self._ml_task_pool.acquire()
            self.queues.send_inputs(self.mpnn_paths, chunk, topic='screen', method='evaluate_mpnn', keep_inputs=True,
                                    task_info={'chunk': self.n_ml_tasks, 'chunk_size': len(chunk)})

            # Mark that we submitted another batch
            self.n_ml_tasks += 1
            
        # Mark that we are done
        self.logger.info('Submitted all molecules to inference tasks')
        self._ml_tasks_submitted.set()
    
    @agent
    def ml_task_consumer(self):
        # Initial list of molecules and their values
        best_mols = []
        best_energies = []
        
        # Loop until all tasks have been received
        n_received = 0
        while not (self._ml_tasks_submitted.is_set() and n_received == self.n_ml_tasks):
            # Receive a task
            result = self.queues.get_result(topic='screen')
            
            # Mark that it was received and another can be submitted
            self._ml_task_pool.release()
            n_received += 1
            self.logger.info(f'Marked result {n_received}/'
                             f'{self.n_ml_tasks if self._ml_tasks_submitted.is_set() else "?"} as received')
            
            # Save the inference result
            self._write_result(result, 'inference_records.jsonld', keep_outputs=False, keep_inputs=False)
            
            # Find the best molecules
            new_mols = result.args[1]
            new_energies = result.value.mean(axis=1)
            
            total_mols = np.hstack((best_mols, new_mols))
            total_energies = np.hstack((best_energies, new_energies))
            
            best_inds = np.argsort(total_energies)[:self.queue_length]
            best_mols = total_mols[best_inds]
            best_energies = total_energies[best_inds]
            self.logger.info(f'Finished updating list to {len(best_mols)} best molecules')
            
        # We are done ranking all of the molecules, time to submit them!
        for i, (s, e) in enumerate(zip(best_mols, best_energies)):
            # Submit a new QC task (but not more than prescribed amount)
            self._qc_task_pool.acquire()
            self.queues.send_inputs(s, topic='simulate', method='compute_atomization_energy', keep_inputs=True,
                                    task_info={'pred': e, 'rank': i})
            self.logger.info(f'Submitted {i}/{len(best_mols)}: {s} with a predicted value of {e}')


if __name__ == '__main__':
    # User inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--redishost", default="127.0.0.1",
                        help="Address at which the redis server can be reached")
    parser.add_argument("--redisport", default="6379",
                        help="Port on which redis is available")
    parser.add_argument('--mpnn-config-directory', help='Directory containing the MPNN-related JSON files',
                        required=True)
    parser.add_argument('--mpnn-model-files', nargs="+", help='Path to the MPNN h5 files', required=True)
    parser.add_argument('--search-space', help='Path to molecules to be screened', required=True)
    parser.add_argument('--qc-spec', help='Name of the QC specification', required=True,
                        choices=['normal_basis', 'xtb', 'small_basis'])
    parser.add_argument('--qc-parallelism', help='Degree of parallelism for QC tasks. For NWChem, number of nodes per task.'
                        ' For XTB, number of tasks per node.', default=1, type=int)
    parser.add_argument("--parallel-guesses", default=1, type=int,
                        help="Number of calculations to maintain in parallel")
    parser.add_argument("--search-size", default=1000, type=int,
                        help="Number of new molecules to evaluate during this search")
    parser.add_argument("--molecules-per-ml-task", default=10000, type=int,
                        help="Number molecules per inference task")
    parser.add_argument("--sampling-fraction", default=None, type=float,
                        help="Fraction of search space to evaluate")
    parser.add_argument("--ml-prefetch", default=0, help="Number of ML tasks to prefech on each node", type=int)
    parser.add_argument("--ml-excess-queue", default=0, type=int,
                        help="Number of tasks to keep in Colmena work queue beyond what would fill the workers.")

    # Parse the arguments
    args = parser.parse_args()
    run_params = args.__dict__
    
    # Define the compute setting for the system (only relevant for NWChem)
    nnodes = int(os.environ.get("COBALT_JOBSIZE", "1"))
    compute_config = {'nnodes': args.qc_parallelism, 'cores_per_rank': 2}
    
    # Determine the number of QC workers and threads per worker
    if args.qc_spec == "xtb":
        qc_workers = nnodes * args.qc_parallelism
        compute_config["ncores"] = 64 // args.qc_parallelism
    else:
        qc_workers = nnodes // args.qc_parallelism
    run_params["nnodes"] = nnodes
    run_params["qc_workers"] = qc_workers
    
    # Load in the models, initial dataset, agent and search space
    with open(os.path.join(args.mpnn_config_directory, 'atom_types.json')) as fp:
        atom_types = json.load(fp)
    with open(os.path.join(args.mpnn_config_directory, 'bond_types.json')) as fp:
        bond_types = json.load(fp)

    # Get QC specification
    qc_spec, code = get_qcinput_specification(args.qc_spec)
    if args.qc_spec != "xtb":
        qc_spec.keywords["dft__iterations"] = 150
        qc_spec.keywords["geometry__noautoz"] = True
    ref_energies = lookup_reference_energies(args.qc_spec)

    # Create an output directory with the time and run parameters
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    out_dir = os.path.join('runs', f'{args.qc_spec}-{start_time.strftime("%d%b%y-%H%M%S")}-{params_hash}')
    os.makedirs(out_dir, exist_ok=True)

    # Save the run parameters to disk
    run_params['version'] = 'simple'
    with open(os.path.join(out_dir, 'run_params.json'), 'w') as fp:
        json.dump(run_params, fp, indent=2)
    with open(os.path.join(out_dir, 'qc_spec.json'), 'w') as fp:
        print(qc_spec.json(), file=fp)
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
    if args.qc_spec == "xtb":
        config = theta_xtb_config(os.path.join(out_dir, 'run-info'), xtb_per_node=args.qc_parallelism, ml_tasks_per_node=1)
    else:
        # ML nodes: N for updating models, 1 for MolDQN, 1 for inference runs
        config = theta_nwchem_config(os.path.join(out_dir, 'run-info'), nodes_per_nwchem=args.qc_parallelism,
                                     ml_prefetch=args.ml_prefetch)

    # Save Parsl configuration
    with open(os.path.join(out_dir, 'parsl_config.txt'), 'w') as fp:
        print(str(config), file=fp)

    # Connect to the redis server
    client_queues, server_queues = make_queue_pairs(args.redishost, args.redisport,
                                                    serialization_method="pickle",
                                                    topics=['simulate', 'screen'],
                                                    keep_inputs=False)

    # Apply wrappers to functions to affix static settings
    #  Update wrapper changes the __name__ field, which is used by the Method Server
    #  TODO (wardlt): Have users set the method name explicitly
    my_compute_atomization = partial(compute_atomization_energy, compute_hessian=args.qc_spec != "xtb",
                                     qc_config=qc_spec, reference_energies=ref_energies,
                                     compute_config=compute_config, code=code)
    my_compute_atomization = update_wrapper(my_compute_atomization, compute_atomization_energy)

    my_evaluate_mpnn = partial(evaluate_mpnn, atom_types=atom_types, bond_types=bond_types, 
                               batch_size=512, n_jobs=64)
    my_evaluate_mpnn = update_wrapper(my_evaluate_mpnn, evaluate_mpnn)

    # Create the method server and task generator
    ml_cfg = {'executors': ['ml']}
    dft_cfg = {'executors': ['qc']}
    doer = ParslMethodServer([(my_evaluate_mpnn, ml_cfg), (my_compute_atomization, dft_cfg)],
                             server_queues, config)
    
    # Compute the number of excess tasks
    excess_tasks = nnodes * args.ml_prefetch + args.ml_excess_queue

    # Configure the "thinker" application
    thinker = Thinker(client_queues,
                      args.search_space,
                      args.mpnn_model_files,
                      out_dir,
                      qc_workers,
                      nnodes,
                      args.molecules_per_ml_task,
                      args.search_size,
                      args.sampling_fraction,
                      excess_tasks)
    logging.info('Created the method server and task generator')

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
