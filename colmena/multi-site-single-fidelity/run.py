from threading import Event, Lock
from typing import Dict, List, Tuple
from functools import partial, update_wrapper
from pathlib import Path
from datetime import datetime
from queue import Queue
import argparse
import logging
import hashlib
import json
import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import proxystore as ps
from rdkit import Chem
from funcx import FuncXClient
from tqdm import tqdm
from colmena.task_server.funcx import FuncXTaskServer
from colmena.models import Result
from colmena.redis.queue import ClientQueues, make_queue_pairs
from colmena.thinker import BaseThinker, result_processor, event_responder, task_submitter
from colmena.thinker.resources import ResourceCounter
from qcelemental.models import OptimizationResult, AtomicResult

from moldesign.score.mpnn import evaluate_mpnn, update_mpnn, retrain_mpnn, MPNNMessage, custom_objects
from moldesign.store.models import MoleculeData
from moldesign.store.recipes import apply_recipes
from moldesign.utils import get_platform_info


def run_simulation(smiles: str, n_nodes: int, spec: str = 'small_basis') -> Tuple[List[OptimizationResult], List[AtomicResult]]:
    """Run the ionization potential computation

    Args:
        smiles: SMILES string to evaluate
        n_nodes: Number of nodes to use
        spec: Name of the quantum chemistry specification
    Returns:
        Relax records for the neutral and ionized geometry
    """
    from moldesign.simulate.functions import generate_inchi_and_xyz, relax_structure
    from moldesign.simulate.specs import get_qcinput_specification

    # Make the initial geometry
    inchi, xyz = generate_inchi_and_xyz(smiles)

    # Make the compute spec
    compute_config = {'nnodes': n_nodes, 'cores_per_rank': 2}

    # Get the specification and make it more resilient
    spec, code = get_qcinput_specification(spec)
    if code == "nwchem":
        spec.keywords["dft__iterations"] = 150
        spec.keywords["geometry__noautoz"] = True

    # Compute the neutral geometry and hessian
    neutral_xyz, _, neutral_relax = relax_structure(xyz, spec, compute_config=compute_config, charge=0, code=code)

    # Compute the relaxed geometry
    oxidized_xyz, _, oxidized_relax = relax_structure(neutral_xyz, spec, compute_config=compute_config, charge=1, code=code)
    return [neutral_relax, oxidized_relax], []


class Thinker(BaseThinker):
    """ML-enhanced optimization loop for molecular design"""

    def __init__(self, queues: ClientQueues,
                 database: List[MoleculeData],
                 search_space: Path,
                 n_to_evaluate: int,
                 n_complete_before_retrain: int,
                 retrain_from_initial: bool,
                 mpnns: List[tf.keras.Model],
                 inference_chunk_size: int,
                 num_qc_workers: int,
                 qc_specification: str,
                 output_dir: str,
                 beta: float,
                 pause_during_update: bool,
                 ps_names: Dict[str, str]):
        """
        Args:
            queues: Queues used to communicate with the method server
            database: Link to the MongoDB instance used to store results
            search_space: Path to a search space of molecules to evaluate
            n_complete_before_retrain: Number of simulations to complete before retraining
            retrain_from_initial: Whether to update the model or retrain it from initial weights
            mpnns: List of MPNNs to use for selecting samples
            output_dir:
            pause_during_update: Whether to stop submitting tasks while task list is updating
            ps_names: mapping of topic to proxystore backend to use (or None if not using ProxyStore)
        """
        super().__init__(queues, ResourceCounter(num_qc_workers, ['simulation']), daemon=True)

        # Configuration for the run
        self.inference_chunk_size = inference_chunk_size
        self.n_complete_before_retrain = n_complete_before_retrain
        self.n_evaluated = 0
        self.retrain_from_initial = retrain_from_initial
        self.mpnns = mpnns.copy()
        self.output_dir = Path(output_dir)
        self.beta = beta
        self.ps_names = ps_names
        self.pause_during_update = pause_during_update

        # Get the name of the property given the specification
        if qc_specification == 'xtb':
            self.property_name = 'xtb-vacuum'
        elif qc_specification == 'small_basis':
            self.property_name = 'smb-vacuum-no-zpe'

        # Get the initial database
        self.database = database
        self.logger.info(f'Populated an initial database of {len(self.database)} entries')

        # Get the target database size
        self.n_to_evaluate = n_to_evaluate

        # List the molecules that have already been searched
        self.already_searched = set([d.identifier['inchi'] for d in self.database])

        # Prepare search space
        self.mols = pd.read_csv(search_space)
        self.mols['dict'] = self.mols['dict'].apply(json.loads)
        self.inference_chunks = np.array_split(self.mols, len(self.mols) // self.inference_chunk_size + 1)
        self.logger.info(f'Split {len(self.mols)} molecules into {len(self.inference_chunks)} chunks for inference')
        
        # Batch add inference chunks to ProxyStore, as they will be used frequently
        inference_msgs = [chunk['dict'].tolist() for chunk in self.inference_chunks]
        if self.ps_names['infer'] is not None:
            keys = [f'search-{mid}' for mid in range(len(inference_msgs))]
            self.inference_proxies = ps.store.get_store(self.ps_names['infer']).proxy_batch(inference_msgs, keys=keys, strict=True)

        # Inter-thread communication stuff
        self.start_inference = Event()  # Mark that inference should start
        self.start_training = Event()  # Mark that retraining should start
        self.task_queue_ready = Event()  # Mark that the task queue is ready
        self.update_in_progress = Event()  # Mark that we are currently re-training the model
        self.update_complete = Event()  # Mark that the update event has finished
        self.task_queue = []  # Holds a list of tasks to be simulated
        self.task_queue_lock = Lock()  # Ensures only one thread edits task queue at a time
        self.ready_models = Queue()
        self.num_training_complete = 0  # Tracks when we are done with training all models
        self.inference_batch = 0

        # Start with inference. Push all models immediately to the queue after triggering inference engine
        self.start_inference.set()
        for mpnn in self.mpnns:
            self.ready_models.put(mpnn)
        
        # Allocate all nodes that are under controlled use to simulation
        self.rec.reallocate(None, 'simulation', 'all')

    @task_submitter(task_type='simulation')
    def submit_qc(self):

        # Wait for the first set of tasks to be available
        self.task_queue_ready.wait()
        
        # If desired, wait until model update is done
        if self.pause_during_update:
            if self.update_in_progress.is_set():
                self.logger.info(f'Waiting until task queue is updated.')
            self.update_complete.wait()

        # Submit the next task
        with self.task_queue_lock:
            inchi, info = self.task_queue.pop(0)
            mol = Chem.MolFromInchi(inchi)
            smiles = Chem.MolToSmiles(mol)
            self.logger.info(f'Submitted {smiles} to simulate with NWChem. Run score: {info["ucb"]}')
            self.already_searched.add(inchi)
            self.queues.send_inputs(smiles, task_info=info,
                                    method='run_simulation', keep_inputs=True,
                                    topic='simulate')

    @result_processor(topic='simulate')
    def process_outputs(self, result: Result):
        # Get basic task information
        smiles, = result.args

        # Release nodes for use by other processes
        self.rec.release("simulation", 1)

        # If successful, add to the database
        if result.success:
            # Mark that we've had another complete result
            self.n_evaluated += 1
            self.logger.info(f'Success! Finished screening {self.n_evaluated}/{self.n_to_evaluate} molecules')

            # Determine whether to start re-training
            if self.n_evaluated % self.n_complete_before_retrain == 0:
                if self.update_in_progress.is_set():
                    self.logger.info(f'Waiting until previous training run completes.')
                else:
                    self.logger.info(f'Starting retraining.')
                    self.start_training.set()
            self.logger.info(f'{self.n_complete_before_retrain - self.n_evaluated % self.n_complete_before_retrain} results needed until we re-train again')

            # Store the data in a molecule data object
            data = MoleculeData.from_identifier(smiles=smiles)
            opt_records, hess_records = result.value
            for r in opt_records:
                data.add_geometry(r)
            for r in hess_records:
                data.add_single_point(r)
            data.update_thermochem()
            apply_recipes(data)
            
            # Add the IPs to the result object
            result.task_info["ip"] = data.oxidation_potential.copy()

            # Add to database
            with open(self.output_dir.joinpath('moldata-records.json'), 'a') as fp:
                print(json.dumps([datetime.now().timestamp(), data.json()]), file=fp)
            self.database.append(data)

            # Write to disk
            with open(self.output_dir.joinpath('qcfractal-records.json'), 'a') as fp:
                for r in opt_records + hess_records:
                    print(r.json(), file=fp)
            self.logger.info(f'Added complete calculation for {smiles} to database.')
            
            # Mark that we've completed one
            if self.n_evaluated >= self.n_to_evaluate:
                self.logger.info(f'No more molecules left to screen')
                self.done.set()
        else:
            self.logger.info(f'Computations failed for {smiles}. Check JSON file for stacktrace')

        # Write out the result to disk
        with open(self.output_dir.joinpath('simulation-results.json'), 'a') as fp:
            print(result.json(exclude={'value'}), file=fp)

    @event_responder(event_name='start_training')
    def train_models(self):
        """Train machine learning models"""
        self.logger.info('Started retraining')

        # Set that a retraining event is in progress
        self.update_complete.clear()
        self.update_in_progress.set()
        self.num_training_complete = 0

        # Save the models as pickle-able messages
        model_msgs = [MPNNMessage(model) for model in self.mpnns]

        # If desired store them as proxies in a large batch
        if self.ps_names['train'] is not None:
            keys = [f'model-{mid}' for mid in range(len(self.mpnns))]
            model_msgs = ps.store.get_store(self.ps_names['train']).proxy_batch(model_msgs, keys=keys, strict=True)
        
        for mid, (model, model_msg) in enumerate(zip(self.mpnns, model_msgs)):
            # Make the database
            train_data = dict(
                (d.identifier['smiles'], d.oxidation_potential[self.property_name])
                for d in self.database
                if self.property_name in d.oxidation_potential
            )

            # Make the MPNN message
            if self.retrain_from_initial:
                self.queues.send_inputs(model.get_config(), train_data, method='retrain_mpnn', topic='train',
                                        task_info={'model_id': mid, 'molecules': list(train_data.keys())},
                                        keep_inputs=False,
                                        input_kwargs={'random_state': mid})
            else:
                self.queues.send_inputs(model_msg, train_data, method='update_mpnn', topic='train',
                                        task_info={'model_id': mid, 'molecules': list(train_data.keys())},
                                        keep_inputs=False,
                                        input_kwargs={'random_state': mid})
            self.logger.info(f'Submitted model {mid} to train with {len(train_data)} entries')

    @result_processor(topic='train')
    def update_weights(self, result: Result):
        """Process the results of the saved model"""

        # Save results to disk
        with open(self.output_dir.joinpath('training-results.json'), 'a') as fp:
            print(result.json(exclude={'inputs', 'value'}), file=fp)

        # Make sure the run completed
        model_id = result.task_info['model_id']
        if not result.success:
            self.logger.warning(f'Training failed for {model_id}')
        else:
            # Update weights
            weights, history = result.value
            self.mpnns[model_id].set_weights(weights)

            # Print out some status info
            self.logger.info(f'Model {model_id} finished training.')
            with open(self.output_dir.joinpath('training-history.json'), 'a') as fp:
                print(repr(history), file=fp)
                
        # Send the model to inference
        self.start_inference.set()
        self.ready_models.put(self.mpnns[model_id])

        # Mark that a model has finished training and trigger inference if all done
        self.num_training_complete += 1
        self.logger.info(f'{len(self.mpnns) - self.num_training_complete} left to go')

    @event_responder(event_name='start_inference')
    def launch_inference(self):
        """Submit inference tasks for the yet-unlabelled samples"""

        self.logger.info('Beginning to submit inference tasks')
        # Make a folder for the models
        model_folder = self.output_dir.joinpath('models')
        model_folder.mkdir(exist_ok=True)
            
        # Submit the chunks to the workflow engine
        for mid in range(len(self.mpnns)):
            # Get a model that is ready for inference
            model = self.ready_models.get()
            
            # Convert it to a pickle-able message
            model_msg = MPNNMessage(model)
            
            # Proxy it once, to be used by all inference tasks
            model_msg_proxy = ps.store.get_store(self.ps_names['infer']).proxy(model_msg, key=f'model-{mid}-{self.inference_batch}')
            
            # Run inference with all segements available
            for cid, (chunk, chunk_msg) in enumerate(zip(self.inference_chunks, self.inference_proxies)):
                self.queues.send_inputs([model_msg_proxy], chunk_msg,
                                        topic='infer', method='evaluate_mpnn',
                                        keep_inputs=False,
                                        task_info={'chunk_id': cid, 'chunk_size': len(chunk), 'model_id': mid})
        self.logger.info('Finished submitting molecules for inference')

    @event_responder(event_name='start_inference')
    def selector(self):
        """Re-prioritize the machine learning tasks"""

        #  Make arrays that will hold the output results from each run
        y_pred = [np.zeros((len(x), len(self.mpnns)), dtype=np.float32) for x in self.inference_chunks]

        # Collect the inference runs
        n_tasks = len(self.inference_chunks) * len(self.mpnns)
        for i in range(n_tasks):
            # Wait for a result
            result = self.queues.get_result(topic='infer')
            self.logger.info(f'Received inference task {i + 1}/{n_tasks}')

            # Save the inference information to disk
            with open(self.output_dir.joinpath('inference-results.json'), 'a') as fp:
                print(result.json(exclude={'value'}), file=fp)

            # Raise an error if this task failed
            if not result.success:
                raise ValueError(f'Inference failed: {result.failure_info.exception}. Check the logs for further details')

            # Store the outputs
            chunk_id = result.task_info.get('chunk_id')
            model_id = result.task_info.get('model_id')
            y_pred[chunk_id][:, model_id] = np.squeeze(result.value)

        # Compute the mean and std for each prediction
        y_pred = np.concatenate(y_pred, axis=0)
        self._select_molecules(y_pred)

        # Mark that inference is complete
        self.inference_batch += 1

        # Mark that the task list has been updated
        self.update_in_progress.clear()
        self.update_complete.set()
        self.task_queue_ready.set()

    def _select_molecules(self, y_pred):
        """Select a list of molecules given the predictions from each model

        Adds them to the task queue

        Args:
            y_pred: List of predictions for each molecule in self.search_space
        """
        # Compute the average and std of predictions
        y_mean = y_pred.mean(axis=1)
        y_std = y_pred.std(axis=1)

        # Rank compounds according to the upper confidence bound
        molecules = self.mols['inchi'].values
        ucb = y_mean + self.beta * y_std
        sort_ids = np.argsort(-ucb)
        best_list = list(zip(molecules[sort_ids].tolist(),
                             y_mean[sort_ids], y_std[sort_ids], ucb[sort_ids]))

        # Push the list to the task queue
        with self.task_queue_lock:
            self.task_queue = []
            for mol, mean, std, ucb in best_list:
                # Add it to list if not in database or not already in queue
                if mol not in self.already_searched and mol not in self.task_queue:
                    # Note: converting to float b/c np.float32 is not JSON serializable
                    self.task_queue.append((mol, {'mean': float(mean), 'std': float(std), 'ucb': float(ucb),
                                                  'batch': self.inference_batch}))

                # If we reach the target number of molecules, stop adding more
                if len(self.task_queue) >= self.n_to_evaluate:
                    break
        self.logger.info('Updated task list')


if __name__ == '__main__':
    # User inputs
    parser = argparse.ArgumentParser()

    # Network configuration details
    group = parser.add_argument_group(title='Network Configuration', description='How to connect to the Redis task queues and task servers, etc')
    group.add_argument("--redishost", default="127.0.0.1", help="Address at which the redis server can be reached")
    group.add_argument("--redisport", default="6379", help="Port on which redis is available")
    
    # Computational infrastructure information
    group = parser.add_argument_group(title='Compute Infrastructure', description='Information about how to run the tasks')
    group.add_argument("--ml-endpoint", required=True, help='FuncX endpoint ID for model training and interface')
    group.add_argument("--qc-endpoint", required=True, help='FuncX endpoint ID for quantum chemistry')
    group.add_argument("--nodes-per-task", default=1, help='Number of nodes per quantum chemistry task. Only needed for NWChem', type=int)
    group.add_argument("--num-qc-workers", required=True, type=int, help="Total number of quantum chemistry workers.")
    group.add_argument("--molecules-per-ml-task", default=10000, type=int, help="Number of molecules per inference chunk")

    # Problem configuration
    group = parser.add_argument_group(title='Problem Definition', description='Defining the search space, models and optimizers-related settings')
    group.add_argument('--qc-specification', default='xtb', choices=['xtb', 'small_basis'],
                       help='Which level of quantum chemistry to run')
    group.add_argument('--mpnn-model-files', nargs="+", help='Path to the MPNN h5 files', required=True)
    group.add_argument('--training-set', help='Path to the molecules used to train the initial models', required=True)
    group.add_argument('--search-space', help='Path to molecules to be screened', required=True)
    group.add_argument("--search-size", default=1000, type=int, help="Number of new molecules to evaluate during this search")
    group.add_argument('--retrain-frequency', default=8, type=int, help="Number of completed computations that will trigger a retraining")
    group.add_argument("--beta", default=1, help="Degree of exploration for active learning. This is the beta from the UCB acquistion function", type=float)
    group.add_argument("--pause-during-update", action='store_true', help='Whether to stop running simulations while updating task list')

    # Parameters related to model retraining
    group = parser.add_argument_group(title='Model Training', description='Settings related to model retraining')
    group.add_argument('--retrain-from-scratch', action='store_true', help='Whether to re-initialize weights before training')
    group.add_argument("--learning-rate", default=1e-3, help="Learning rate for re-training the models", type=float)
    group.add_argument('--num-epochs', default=512, type=int, help='Maximum number of epochs for the model training')

    # Parameters related to ProxyStore
    group = parser.add_argument_group(title='ProxyStore', description='Settings related to ProxyStore')
    group.add_argument('--simulate-ps-backend', default=None, choices=[None, 'redis', 'file', 'globus'], help='ProxyStore backend to use with "simulate" topic')
    group.add_argument('--infer-ps-backend', default=None, choices=[None, 'redis', 'file', 'globus'], help='ProxyStore backend to use with "infer" topic')
    group.add_argument('--train-ps-backend', default=None, choices=[None, 'redis', 'file', 'globus'], help='ProxyStore backend to use with "train" topic')
    group.add_argument('--ps-threshold', default=500000, type=int, help='Min size in bytes for transferring objects via ProxyStore')
    group.add_argument('--ps-file-dir', default=None, help='Filesystem directory to use with the ProxyStore file backend')
    group.add_argument('--ps-globus-config', default=None, help='Globus Endpoint config file to use with the ProxyStore Globus backend')

    # Parse the arguments
    args = parser.parse_args()
    run_params = args.__dict__

    # Load in the models, initial dataset, agent and search space
    models = [tf.keras.models.load_model(path, custom_objects=custom_objects) for path in tqdm(args.mpnn_model_files, desc='Loading models')]

    # Read in the training set
    with open(args.training_set) as fp:
        database = [MoleculeData.parse_raw(line) for line in fp]

    # Create an output directory with the time and run parameters
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    out_dir = os.path.join('runs', f'{args.qc_specification}-N{args.num_qc_workers}-n{args.nodes_per_task}-{params_hash}-{start_time.strftime("%d%b%y-%H%M%S")}')
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
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, handlers=handlers)
    logging.info(f'Run directory: {out_dir}')
    
    # Make the PS files directory inside of the run directory
    
    ps_file_dir = os.path.abspath(os.path.join(out_dir, args.ps_file_dir))
    os.makedirs(ps_file_dir, exist_ok=False)
    logging.info(f'Scratch directory for ProxyStore files: {ps_file_dir}')

    # Init ProxyStore backends
    ps_backends = {args.simulate_ps_backend, args.infer_ps_backend, args.train_ps_backend}
    if 'redis' in ps_backends:
        ps.store.init_store(ps.store.STORES.REDIS, name='redis', hostname=args.redishost, port=args.redisport)
    if 'file' in ps_backends:
        if args.ps_file_dir is None:
            raise ValueError('Must specify --ps-file-dir to use the filesystem ProxyStore backend')
        ps.store.init_store(ps.store.STORES.FILE, name='file', store_dir=ps_file_dir)
    if 'globus' in ps_backends:
        if args.ps_globus_config is None:
            raise ValueError('Must specify --ps-globus-config to use the Globus ProxyStore backend')
        endpoints = ps.store.globus.GlobusEndpoints.from_json(args.ps_globus_config)
        ps.store.init_store(ps.store.STORES.GLOBUS, name='globus', endpoints=endpoints)
    ps_names = {'simulate': args.simulate_ps_backend, 'infer': args.infer_ps_backend, 'train': args.train_ps_backend}

    # Connect to the redis server
    client_queues, server_queues = make_queue_pairs(args.redishost,
                                                    name=start_time.strftime("%d%b%y-%H%M%S"),
                                                    port=args.redisport,
                                                    topics=['simulate', 'infer', 'train'],
                                                    serialization_method='pickle',
                                                    keep_inputs=True,
                                                    proxystore_name=ps_names,
                                                    proxystore_threshold=args.ps_threshold)

    # Apply wrappers to functions to affix static settings
    #  Update wrapper changes the __name__ field, which is used by the Method Server
    my_evaluate_mpnn = partial(evaluate_mpnn, batch_size=128, cache=True)
    my_evaluate_mpnn = update_wrapper(my_evaluate_mpnn, evaluate_mpnn)

    my_update_mpnn = partial(update_mpnn, num_epochs=args.num_epochs, learning_rate=args.learning_rate, bootstrap=True)
    my_update_mpnn = update_wrapper(my_update_mpnn, update_mpnn)

    my_retrain_mpnn = partial(retrain_mpnn, num_epochs=args.num_epochs, learning_rate=args.learning_rate, bootstrap=True)
    my_retrain_mpnn = update_wrapper(my_retrain_mpnn, retrain_mpnn)

    my_run_simulation = partial(run_simulation, n_nodes=args.nodes_per_task, spec=args.qc_specification)
    my_run_simulation = update_wrapper(my_run_simulation, run_simulation)

    # Create the task servers
    fx_client = FuncXClient()
    task_map = dict((f, args.ml_endpoint) for f in [my_evaluate_mpnn, my_update_mpnn, my_retrain_mpnn])
    task_map[my_run_simulation] = args.qc_endpoint
    doer = FuncXTaskServer(task_map, fx_client, server_queues)

    # Configure the "thinker" application
    thinker = Thinker(client_queues,
                      database,
                      args.search_space,
                      args.search_size,
                      args.retrain_frequency,
                      args.retrain_from_scratch,
                      models,
                      args.molecules_per_ml_task,
                      args.num_qc_workers,
                      args.qc_specification,
                      out_dir,
                      args.beta,
                      args.pause_during_update,
                      ps_names)
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

    # Cleanup ProxyStore backends (i.e., delete objects on the filesystem
    # for file/globus backends)
    for ps_backend in ps_backends:
        if ps_backend is not None:
            ps.store.get_store(ps_backend).cleanup()
