"""Multi-site version of the application"""

from typing import List, Dict, Union, Tuple
from functools import partial, update_wrapper
from urllib import parse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import argparse
import logging
import hashlib
import json
import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from colmena.task_server import ParslTaskServer
from colmena.redis.queue import ClientQueues, make_queue_pairs
from colmena.thinker import event_responder

from config import theta_persistent
from moldesign.score.mpnn import evaluate_mpnn, update_mpnn, retrain_mpnn
from moldesign.score.schnet import train_schnet, evaluate_schnet
from moldesign.store.models import RedoxEnergyRecipe
from moldesign.store.mongo import MoleculePropertyDB
from moldesign.store.recipes import get_recipe_by_name
from moldesign.utils import get_platform_info
from sim import compute_adiabatic, compute_vertical, compute_single_point
from run import Thinker

# Disable all GPUs for the planning process
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'


# Entry for the priority queue
@dataclass(order=True)
class _PriorityEntry:
    score: float
    item: dict = field(compare=False)


class MultiThinker(Thinker):
    """ML-enhanced optimization loop for molecular design"""

    def __init__(self, queues: ClientQueues,
                 num_qc_workers: int,
                 num_ml_workers: int,
                 database: MoleculePropertyDB,
                 search_space: List[str],
                 n_to_evaluate: int,
                 n_complete_before_reorder: int,
                 n_complete_before_retrain: int,
                 retrain_from_initial: bool,
                 models: Dict[str, List[Path]],
                 calibration_factor: Dict[str, float],
                 inference_chunk_size: int,
                 nodes_per_qc: int,
                 output_dir: Union[str, Path],
                 beta: float,
                 random_seed: int,
                 oxidize: bool,
                 target_recipe: RedoxEnergyRecipe,
                 target_range: Tuple[float, float]):
        """
        Args:
            queues: Queues used to communicate with the method server
            num_qc_workers: Number of nodes available for quantum chem computations
            num_ml_workers: Number of nodes available for ML computations
            database: Link to the MongoDB instance used to store results
            search_space: List of InChI strings which define the search space
            n_complete_before_reorder: Number of simulations to complete before re-running inference
            n_complete_before_retrain: Number of simulations to complete before retraining
            retrain_from_initial: Whether to update the model or retrain it from initial weights
            models: Path to models used for inference. Keys are the name of the computation that
                each type of model is used to decide whether to run for a molecule:
                - `vertical` are MPNN models that determine if we should run an initial, vertical redox;
                - `adiabatic` are SchNet models that determine if we should run an adiabatic redox with a small basis set
                - `normal` are SchNet models that determine if we should re-compute energies with a normal basis set
            calibration_factor: Factor used to adjust the uncertainties
            inference_chunk_size: Maximum number of molecules per inference task
            nodes_per_qc: Number of nodes per QC task
            output_dir: Where to write the output files
            beta: Amount to weight uncertainty in the activation function
            random_seed: Random seed for the model (re)trainings
            target_range: Target range for the property
            oxidize: Whether to compute the oxidation instead of the reduction potential
            target_level: Name of the final level of the property to be computed
        """

        super(MultiThinker, self).__init__(
            queues=queues,
            num_workers=num_ml_workers + num_qc_workers,
            database=database,
            search_space=search_space,
            n_to_evaluate=n_to_evaluate,
            n_complete_before_reorder=n_complete_before_reorder,
            n_complete_before_retrain=n_complete_before_retrain,
            retrain_from_initial=retrain_from_initial,
            models=models,
            calibration_factor=calibration_factor,
            inference_chunk_size=inference_chunk_size,
            nodes_per_qc=nodes_per_qc,
            output_dir=output_dir,
            beta=beta,
            random_seed=random_seed,
            oxidize=oxidize,
            target_recipe=target_recipe,
            target_range=target_range
        )
        self.num_ml_workers = num_ml_workers
        self.num_qc_workers = num_qc_workers

        # Move required nodes from "inference" (original __init__ puts all on inference) to simulation
        self.rec.reallocate('inference', 'simulation', num_qc_workers)

    @event_responder(event_name='start_inference', reallocate_resources=True, max_slots='num_qc_workers',
                     gather_from='training', gather_to='inference', disperse_to='training')
    def launch_inference(self):
        """Submit inference tasks for the yet-unlabelled samples"""
        self._launch_inference()

    @event_responder(event_name='start_training', reallocate_resources=True, max_slots='num_qc_workers',
                     gather_from='simulation', gather_to='training', disperse_to='inference')
    def train_models(self):
        self._train_model()


if __name__ == '__main__':
    # User inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--redishost", default="127.0.0.1",
                        help="Address at which the redis server can be reached")
    parser.add_argument("--redisport", default="6379",
                        help="Port on which redis is available")

    # Model-related configuration
    parser.add_argument('--vertical-model-files', nargs="+", help='Path to the MPNN h5 files', required=True)
    parser.add_argument('--vertical-calibration', default=1, help='Calibration for MPNN uncertainties', type=float)

    parser.add_argument('--adiabatic-model-files', nargs="+", help='Path to SchNet best_model files', required=True)
    parser.add_argument('--adiabatic-calibration', default=1, help='Calibration for SchNet uncertainties', type=float)

    parser.add_argument('--normal-model-files', nargs="+", help='Path to SchNet best_model files', required=True)
    parser.add_argument('--normal-calibration', default=1, help='Calibration for SchNet uncertainties', type=float)

    parser.add_argument("--learning-rate", default=1e-3, help="Initial learning rate for re-training the models",
                        type=float)
    parser.add_argument('--num-epochs', default=512, type=int, help='Maximum number of epochs for the model training')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size for model training')
    parser.add_argument('--random-seed', default=0, type=int, help='Random seed for model (re)trainings')
    parser.add_argument('--retrain-from-scratch', action='store_true', help='Whether to retrain models from scratch')
    parser.add_argument('--train-timeout', default=None, type=float, help='Timeout for training operation (s)')
    parser.add_argument('--train-patience', default=None, type=int, help='Patience for training operation (epochs)')

    # Molecule data / search space related
    parser.add_argument('--mongo-url', default='mongod://localhost:27017/', help='URL of the mongo server')
    parser.add_argument('--search-space', help='Path to molecules to be screened', required=True)
    parser.add_argument('--max-heavy-atoms', type=int, help='Maximum molecule size')
    parser.add_argument('--search-downselect', help='How many molecules to use from the search space', type=int)
    parser.add_argument('--oxidize', action='store_true', help='Optimize the ionization potential')
    parser.add_argument('--target-level', help='Level of accuracy to compute. Name of a RedoxReceipe')

    # Active-learning related
    parser.add_argument("--search-size", default=200000, type=int,
                        help="Number of new molecules to evaluate during this search")
    parser.add_argument('--target-range', default=(10, np.inf), type=float, nargs=2,
                        help='Target range (min, max) for IP')
    parser.add_argument('--retrain-frequency', default=50, type=int,
                        help="Number of completed high-fidelity computations need to trigger model retraining")
    parser.add_argument('--reorder-frequency', default=50, type=int,
                        help="Number of completed low-fidelity computations "
                             "need to trigger reordering the task list using new data")
    parser.add_argument("--molecules-per-ml-task", default=8192, type=int,
                        help="Number molecules per inference task")
    parser.add_argument("--beta", default=1, help="Degree of exploration for active learning. "
                                                  "This is the beta from the UCB acquisition function", type=float)

    # Execution system related
    parser.add_argument('--nodes-per-task', default=4, type=int,
                        help='Number of nodes per quantum chemistry task')
    parser.add_argument('--num-qc-nodes', default=8, type=int, help='Number of nodes to request for QC computations')
    parser.add_argument('--num-ml-nodes', default=8, type=int, help='Number of nodes to request for ML computations')

    # Parse the arguments
    args = parser.parse_args()
    run_params = args.__dict__

    # Define the compute setting for the system (only relevant for NWChem)
    qc_nodes = args.num_qc_nodes
    ml_nodes = args.num_ml_nodes
    run_params["nnodes"] = qc_nodes
    run_params["qc_workers"] = qc_nodes / args.nodes_per_task

    # Get the desired computational settings
    recipe = get_recipe_by_name(args.target_level)

    # Connect to MongoDB
    mongo_url = parse.urlparse(args.mongo_url)
    mongo = MoleculePropertyDB.from_connection_info(mongo_url.hostname, mongo_url.port)

    # Load in the search space
    full_search = pd.read_csv(args.search_space, delim_whitespace=True)
    if args.max_heavy_atoms is not None:
        full_search.query(f'heavy_atoms <= {args.max_heavy_atoms}', inplace=True)
    if args.search_downselect is not None and len(full_search) > args.search_downselect:
        full_search = full_search.sample(args.search_downselect, random_state=args.random_seed)
    search_space = full_search['inchi'].values

    # Create an output directory with the time and run parameters
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    out_dir = Path('runs').joinpath(f'{args.target_level}-{start_time.strftime("%d%b%y-%H%M%S")}-persistant-{params_hash}')
    out_dir.mkdir(exist_ok=False, parents=True)

    # Save the run parameters to disk
    with open(os.path.join(out_dir, 'run_params.json'), 'w') as fp:
        json.dump(run_params, fp, indent=2)
    with open(os.path.join(out_dir, 'environment.json'), 'w') as fp:
        json.dump(dict(os.environ), fp, indent=2)

    # Save the platform information to disk
    host_info = get_platform_info()
    with open(os.path.join(out_dir, 'host_info.json'), 'w') as fp:
        json.dump(host_info, fp, indent=2)

    # Build summary of the models and model 
    calibration = {}
    models = {}
    for tag in ['vertical', 'adiabatic', 'normal']:
        # Load the user inputs
        models[tag] = run_params[f'{tag}_model_files']
        calibration[tag] = run_params[f'{tag}_calibration']

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
    config = theta_persistent(os.path.join(out_dir, 'run-info'), nodes_per_nwchem=args.nodes_per_task,
                              qc_nodes=qc_nodes, ml_nodes=ml_nodes)

    # Save Parsl configuration
    with open(os.path.join(out_dir, 'parsl_config.txt'), 'w') as fp:
        print(str(config), file=fp)

    # Connect to the redis server
    client_queues, server_queues = make_queue_pairs(args.redishost, args.redisport,
                                                    serialization_method="pickle",
                                                    topics=['simulate', 'infer', 'train'],
                                                    keep_inputs=False)

    # Apply wrappers to functions to affix static settings
    #  Update wrapper changes the __name__ field, which is used by the Method Server
    def _fix_arguments(func, **kwargs):
        my_func = partial(func, **kwargs)
        return update_wrapper(my_func, func)


    my_evaluate_mpnn = _fix_arguments(evaluate_mpnn, batch_size=args.batch_size, cache=True, n_jobs=32)
    my_update_mpnn = _fix_arguments(update_mpnn, num_epochs=args.num_epochs,
                                    learning_rate=args.learning_rate, bootstrap=True, batch_size=args.batch_size,
                                    patience=args.train_patience, timeout=args.train_timeout)
    my_retrain_mpnn = _fix_arguments(retrain_mpnn, num_epochs=args.num_epochs,
                                     learning_rate=args.learning_rate, bootstrap=True, batch_size=args.batch_size,
                                     patience=args.train_patience, timeout=args.train_timeout)
    my_evaluate_schnet = _fix_arguments(evaluate_schnet, property_name='delta', batch_size=args.batch_size)
    my_train_schnet = _fix_arguments(train_schnet, property_name='delta', num_epochs=args.num_epochs,
                                     learning_rate=args.learning_rate, bootstrap=True, batch_size=args.batch_size,
                                     reset_weights=not args.retrain_from_scratch,
                                     patience=args.train_patience, timeout=args.train_timeout)

    my_compute_vertical = _fix_arguments(compute_vertical, spec_name=recipe.geometry_level, n_nodes=args.nodes_per_task)
    my_compute_adiabatic = _fix_arguments(compute_adiabatic, spec_name=recipe.geometry_level,
                                          n_nodes=args.nodes_per_task)
    my_compute_single = _fix_arguments(compute_single_point, n_nodes=args.nodes_per_task)

    # Create the method server and task generator
    ml_cfg = {'executors': ['ml']}
    dft_cfg = {'executors': ['qc']}
    doer = ParslTaskServer([(my_compute_vertical, dft_cfg), (my_compute_adiabatic, dft_cfg),
                            (my_compute_single, dft_cfg),
                            (my_evaluate_mpnn, ml_cfg), (my_evaluate_schnet, ml_cfg),
                            (my_update_mpnn, ml_cfg), (my_retrain_mpnn, ml_cfg), (my_train_schnet, ml_cfg)],
                           server_queues, config)

    # Configure the "thinker" application
    thinker = MultiThinker(client_queues,
                           qc_nodes,
                           ml_nodes,
                           mongo,
                           search_space,
                           args.search_size,
                           args.reorder_frequency,
                           args.retrain_frequency,
                           args.retrain_from_scratch,
                           models,
                           calibration,
                           args.molecules_per_ml_task,
                           args.nodes_per_task,
                           out_dir,
                           args.beta,
                           args.random_seed,
                           args.oxidize,
                           recipe,
                           args.target_range)
    logging.info('Created the task server and task generator')

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
