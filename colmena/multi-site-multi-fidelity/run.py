from dataclasses import dataclass, field
from threading import Event, Lock, Barrier
from typing import Dict, List, Tuple
from functools import partial, update_wrapper
from pathlib import Path
from datetime import datetime
from queue import Queue, PriorityQueue, Empty
import argparse
import logging
import hashlib
import json
import sys
import os

import yaml
import numpy as np
import pandas as pd
import proxystore as ps
from funcx import FuncXClient
from colmena.task_server.funcx import FuncXTaskServer
from colmena.models import Result
from colmena.redis.queue import ClientQueues, make_queue_pairs
from colmena.thinker import BaseThinker, result_processor, event_responder, task_submitter, agent
from colmena.thinker.resources import ResourceCounter
from pydantic.utils import defaultdict

from moldesign.simulate.functions import relax_structure, run_single_point
from moldesign.score.mpnn import evaluate_mpnn, update_mpnn, retrain_mpnn
from moldesign.score.schnet import train_schnet, evaluate_schnet
from moldesign.specify import MultiFidelitySearchSpecification
from moldesign.store.models import MoleculeData
from moldesign.store.mongo import MoleculePropertyDB
from moldesign.store.recipes import apply_recipes, get_recipe_by_name
from moldesign.utils import get_platform_info

from sim import get_relaxation_args, get_single_point_args


@dataclass(order=True)
class _PriorityEntry:
    score: float
    inchi: str = field(compare=False)
    item: dict = field(compare=False)


class Thinker(BaseThinker):
    """ML-enhanced optimization loop for molecular design"""

    def __init__(self, queues: ClientQueues,
                 database: MoleculePropertyDB,
                 search_space: Path,
                 search_space_name: str,
                 n_to_evaluate: int,
                 n_complete_before_retrain: int,
                 retrain_from_initial: bool,
                 inference_chunk_size: int,
                 num_qc_workers: int,
                 search_spec: MultiFidelitySearchSpecification,
                 output_dir: str,
                 beta: float,
                 pause_during_update: bool,
                 ps_names: Dict[str, str]):
        """
        Args:
            queues: Queues used to communicate with the method server
            database: Link to the MongoDB instance used to store results
            search_space: Path to a search space of molecules to evaluate
            search_space_name: Short name for the source of the molecules
            n_complete_before_retrain: Number of simulations to complete before retraining
            retrain_from_initial: Whether to update the model or retrain it from initial weights
            output_dir: Directory in which to write logs
            pause_during_update: Whether to stop submitting tasks while task list is updating
            ps_names: mapping of topic to proxystore backend to use (or None if not using ProxyStore)
        """
        super().__init__(queues, ResourceCounter(num_qc_workers, ['simulation']), daemon=True)

        # Configuration for the run
        self.inference_chunk_size = inference_chunk_size
        self.n_complete_before_retrain = n_complete_before_retrain
        self.n_evaluated = 0
        self.retrain_from_initial = retrain_from_initial
        self.output_dir = Path(output_dir)
        self.beta = beta
        self.ps_names = ps_names
        self.pause_during_update = pause_during_update
        self.search_space = search_space
        self.search_space_name = search_space_name
        self.search_spec = search_spec
        self.num_models = sum([len(x.model_paths) for x in search_spec.model_levels]) \
                          + len(search_spec.base_model.model_paths)

        # Get the initial database
        self.database = database
        self.logger.info(f'Connected to a database of {database.collection.count_documents({})} entries')

        # Get the target database size
        self.n_to_evaluate = n_to_evaluate

        # Inter-thread communication stuff
        self.start_inference = Event()  # Mark that inference should start
        self.start_training = Event()  # Mark that retraining should start
        self.task_queue_ready = Event()  # Mark that the task queue is ready
        self.update_in_progress = Event()  # Mark that we are currently re-training the model
        self.update_complete = Event()  # Mark that the update event has finished
        self.task_queue: Queue[_PriorityEntry] = PriorityQueue()  # Holds a list of tasks to be simulated
        self.next_task: Queue[Tuple[MoleculeData, dict, Tuple]] = Queue(1)  # Get the next task to be run
        self.task_queue_lock = Lock()  # Ensures only one thread edits task queue at a time
        self.ready_models = Queue()
        self.num_training_complete = 0  # Tracks when we are done with training all models
        self.inference_batch = 0
        self.inference_mols: Dict[str, List[str]] = defaultdict(
            list)  # List of molecules run during inference at different levels
        self.inference_results: Dict[str, List[np.ndarray]] = defaultdict(list)  # Placeholder for inference results
        self.inference_ready = Barrier(2)  # When inference tasks are ready to be sent/received
        self.already_ran = set()

        # Start with inference
#         self.start_inference.set()
#         for level in ['base'] + search_spec.levels[:-1]:
#             spec = search_spec.get_models(level)
#             n_models = len(spec.model_paths)
#             for i in range(n_models):
#                 self.ready_models.put((level, i))

        # Start with training so that we ensure the models are as up-to-date as possible
        self.start_training.set()

        # Allocate all nodes that are under controlled use to simulation
        self.rec.reallocate(None, 'simulation', 'all')

    @agent()
    def qc_queuer(self):
        """Gets the next task from the task list"""

        # If desired, wait until model update is done
        if self.pause_during_update:
            if self.update_in_progress.is_set():
                self.logger.info(f'Waiting until task queue is updated.')
            self.update_complete.wait()

        while not self.done.is_set():
            # Get the next task
            next_task = self.task_queue.get()
            self.logger.info(f'Pulled next molecule {next_task.inchi} with priority {next_task.score:.2f}')

            # Get what we know about this molecule already
            record = self.database.get_molecule_record(inchi=next_task.inchi)

            # Determine what to run next
            next_step = self.search_spec.get_next_step(record)
            if next_step is None:
                self.logger.info(f'{next_task.inchi} is already complete. Skipping')
                continue
            next_recipe = get_recipe_by_name(next_step)
            self.logger.info(f'Preparing to run calculations for {next_step}')

            if next_step is None or next_step == self.search_spec.levels[0]:
                previous_recipe = None
            else:
                previous_step_id = self.search_spec.levels.index(next_step) - 1
                previous_recipe = get_recipe_by_name(self.search_spec.levels[previous_step_id])

            # Get the recipes
            try:
                to_run = next_recipe.get_required_calculations(record, self.search_spec.oxidation_state, previous_recipe)
            except ValueError:
                continue
            self.logger.info(f'{len(to_run)} more calculations required to complete {next_step}')

            # Add them to a queue
            n_skipped = 0
            for task in to_run:
                if (record.key, task) not in self.already_ran:
                    self.next_task.put((record, next_task.item, task))
                    self.already_ran.add((record.key, task))
                else:
                    n_skipped += 1
            if n_skipped > 0:
                self.logger.info(f'Skipped submitted {n_skipped} computations that have already been ran')

            # Check if we are done
            if self.n_evaluated >= self.n_to_evaluate:
                self.logger.info(f'No more molecules left to screen')
                break

    @task_submitter(task_type='simulation')
    def submit_qc(self):
        """Submit QC tasks when resources are available"""

        # Get the next task
        record, task_info, (qc_spec, xyz, chg, solvent, is_relax) = self.next_task.get()

        # Launch the appropriate function
        inchi = record.identifier['inchi']
        task_info['inchi'] = inchi
        if is_relax:
            args, kwargs = get_relaxation_args(xyz, charge=chg, spec_name=qc_spec)
            self.queues.send_inputs(
                *args,
                input_kwargs=kwargs,
                method='relax_structure',
                topic='simulate',
                task_info=task_info
            )
        else:
            args, kwargs = get_single_point_args(xyz, chg, solvent, qc_spec)
            self.queues.send_inputs(
                *args,
                input_kwargs=kwargs,
                method='run_single_point',
                topic='simulate',
                task_info=task_info
            )
        self.logger.info(f'Submitted a {"relax" if is_relax else "single_point"} '
                         f'task for {record.identifier["smiles"]} at the {qc_spec} level')

    @result_processor(topic='simulate')
    def process_outputs(self, result: Result):
        # Release nodes for use by other processes
        self.rec.release("simulation", 1)

        # Unpack the task information
        inchi = result.task_info['inchi']
        method = result.method

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
            self.logger.info(f'{self.n_complete_before_retrain - self.n_evaluated % self.n_complete_before_retrain}'
                             ' results needed until we re-train again')

            # Store the data in a molecule data object
            data = self.database.get_molecule_record(inchi=inchi)
            if method == 'relax_structure':
                data.add_geometry(result.value)
            else:
                data.add_single_point(result.value)
            data.update_thermochem()
            apply_recipes(data)

            # Attach the data source for the molecule
            data.subsets.append(self.search_space_name)

            # Add the IPs to the result object
            result.task_info["ip"] = data.oxidation_potential.copy()
            result.task_info["ea"] = data.reduction_potential.copy()

            # Add to database
            with open(self.output_dir.joinpath('moldata-records.json'), 'a') as fp:
                print(json.dumps([datetime.now().timestamp(), data.json()]), file=fp)
            self.database.update_molecule(data)

            # Write to disk
            with open('qcfractal-records.json', 'a') as fp:
                print(result.value.json(), file=fp)
            self.logger.info(f'Added complete calculation for {inchi} to database.')
        else:
            self.logger.info(f'Computations failed for {inchi}. Check JSON file for stacktrace')

        # Write out the result to disk
        with open(self.output_dir.joinpath('simulation-results.json'), 'a') as fp:
            print(result.json(exclude={'inputs', 'value'}), file=fp)
        self.logger.info(f'Processed simulation task.')

    @event_responder(event_name='start_training')
    def train_models(self):
        """Train machine learning models"""
        self.logger.info('Started retraining')

        # Set that a retraining event is in progress
        self.update_complete.clear()
        self.update_in_progress.set()
        self.num_training_complete = 0
        
        # Trigger the inference to begin, so that it gets time to prepare inference chunks
        self.start_inference.set()

        # Start by sending the base models for training
        train_data = self.search_spec.get_base_training_set(self.database)
        self.logger.info(f'Pulled {len(train_data)} molecules to retrain the base model')
        if self.ps_names['train'] is not None:
            train_data = ps.store.get_store(self.ps_names['train']).proxy(train_data, key='train-data-base')

        for mid, model_msg in enumerate(self.search_spec.base_model.load_all_model_messages()):
            self.queues.send_inputs(model_msg.get_model().get_config(), train_data,
                                    method='retrain_mpnn', topic='train',
                                    task_info={'model_id': mid, 'level': 'base'},
                                    keep_inputs=False,
                                    input_kwargs={'random_state': mid})
            self.logger.info(f'Submitted model {mid} to train with {len(train_data)} entries')

        # Next, send each level of the calibration models
        for level_id, level in enumerate(self.search_spec.model_levels):
            # TODO (wardlt): For now, don't retrain these models. Just push them to infernece
#             for model_id in range(len(level.model_paths)):
#                 self.ready_models.put((level.base_fidelity, model_id))
#             continue  # Skips actually training them
            
            # Prepare the training set
            train_data = self.search_spec.get_calibration_training_set(level_id, self.database)
            self.logger.info(f'Pulled {len(train_data)} molecules to retrain {level.base_fidelity} calibrator')
            if self.ps_names['train'] is not None:
                train_data = ps.store.get_store(self.ps_names['train']).proxy(train_data, key=f'train-data-{level_id}')

            # Prepare the model messages
            model_msgs = list(level.load_all_model_messages())
#             if self.ps_names['train'] is not None:
#                 model_msgs = ps.store.get_store(self.ps_names['train']).proxy_batch(
#                     model_msgs, keys=[f'level-{level_id}-model-{i}' for i in range(len(model_msgs))]
#                 )

            for mid, model_msg in enumerate(level.load_all_model_messages()):
                self.queues.send_inputs(model_msg.get_model().get_config(), train_data,
                                        method='retrain_mpnn', topic='train',
                                        task_info={'model_id': mid, 'level': level.base_fidelity},
                                        keep_inputs=False,
                                        input_kwargs={'random_state': mid})
                
        self.logger.info('Finished submitting training')

    @result_processor(topic='train')
    def update_weights(self, result: Result):
        """Process the results of the saved model"""

        # Save results to disk
        with open(self.output_dir.joinpath('training-results.json'), 'a') as fp:
            print(result.json(exclude={'inputs', 'value'}), file=fp)

        # Make sure the run completed
        model_id = result.task_info['model_id']
        level = result.task_info['level']
        if not result.success:
            self.logger.warning(f'Training failed for level {level} model {model_id}')
        else:
            self.logger.info(f'Training succeeded for level {level} model {model_id}')

            # Get the update message
            output = result.value
            assert str(output) != ""  # TODO (wardlt): Figure out if `iter` doesn't work with lazy_object_proxy
            message, history = output

            # Update the model
            if level == 'base':
                model_collection = self.search_spec.base_model 
            else:
                level_id = self.search_spec.levels.index(level)
                model_collection = self.search_spec.model_levels[level_id]
            model_collection.update_model(model_id, message)
            self.logger.info(f'Saved updated model to {model_collection.model_paths[model_id]}')

            # Print out some status info
            with open(self.output_dir.joinpath('training-history.json'), 'a') as fp:
                print(repr(history), file=fp)

        # Send the model to inference
        self.ready_models.put((level, model_id))

        # Mark that a model has finished training and trigger inference if all done
        self.num_training_complete += 1

    @event_responder(event_name='start_inference')
    def launch_inference(self):
        """Submit inference tasks for the yet-unlabelled samples"""

        # Prepare storage for the inference results
        inference_inputs = defaultdict(list)  # Key is the level. Value is whatever is taken as inputs
        initial_value = defaultdict(list)  # Key is the level. Value is the value at the lower level of fidelity
        
         # Load in space of allowed molecules
        mols = pd.read_csv(self.search_space)
        mols.drop_duplicates('inchi', inplace=True)
        allowed_inchis = set(mols['inchi'])
        self.logger.info(f'Read in a dataset of {len(allowed_inchis)} molecules')

        # Store the search space as a dictionary mapping InChI to a parsed version
        #  Make sure it includes only molecules that are not yet in the database
        known_molecules = self.database.get_molecules()
        unlabeled_molecules = [
            (i, d) for i, d in
            zip(mols['inchi'], mols['dict'].apply(json.loads))
            if i not in known_molecules
        ]
        self.logger.info(f'Identified {len(unlabeled_molecules)} molecules yet to be evaluated')

        # Store the unlabeled molecules first
        self.inference_mols['base'].extend(x[0] for x in unlabeled_molecules)
        inference_inputs['base'].extend(x[1] for x in unlabeled_molecules)
        initial_value['base'].extend([0.] * len(inference_inputs['base']))
        self.logger.info(f'Preparing to run inference on {len(initial_value["base"])} molecules')

        # Now add the ones from the database that have not finished computing the target property
        for record in self.database.collection.find({self.search_spec.target_property: {'$exists': False}}):
            record = MoleculeData.parse_obj(record)
            if record.identifier['inchi'] not in allowed_inchis:
                continue
            current_step, inputs, init_value = self.search_spec.get_inference_inputs(record)
            self.inference_mols[current_step].append(record.identifier['inchi'])
            initial_value[current_step].append(init_value)
            inference_inputs[current_step].append(inputs)

        # Send out proxies for the inference inputs in chunks
        inference_msgs = {}
        inference_chunks = {}
        for level, inputs in inference_inputs.items():
            # Determine the number of inference batches to send out
            n_chunks = len(inputs) // self.inference_chunk_size + 1

            # Divide the inputs and output storage into chunks
            inference_chunks[level] = np.array_split(inputs, n_chunks)
            self.inference_results[level] = np.array_split(initial_value[level], n_chunks)

            # Make the proxies, if desirable
            if self.ps_names['infer'] is not None:
                keys = [f'search-{level}-{mid}' for mid in range(len(inference_chunks[level]))]
                inference_msgs[level] = ps.store.get_store(self.ps_names['infer']).proxy_batch(
                    inference_chunks[level], keys=keys, strict=True
                )
            else:
                inference_msgs[level] = inference_chunks[level]

            self.logger.info(f'Prepared to submit {len(inputs)} molecules for {level} in {n_chunks} batches')

        # Submit the chunks to the workflow engine
        self.logger.info(f'Ready to submit inference tasks')
        self.inference_ready.wait()
        for i in range(self.num_models):
            # Get a model that is ready for inference
            level, mid = self.ready_models.get()
            self.logger.info(f'Sending inference requests out for {level} model {mid}')

            # Convert it to a pickle-able message
            model_spec = self.search_spec.get_models(level)
            model_msg = model_spec.load_model_message(mid)

            # Proxy it once, to be used by all inference tasks
            if self.ps_names['infer'] is not None:
                model_msg = ps.store.get_store(self.ps_names['infer']).proxy(model_msg, key=f'model-{level}-{mid}')

            # Run inference with all segments available
            for cid, (chunk, chunk_msg) in enumerate(zip(inference_chunks[level], inference_msgs[level])):
                self.queues.send_inputs([model_msg], chunk_msg,
                                        topic='infer', method=f'evaluate_{model_spec.model_type}',
                                        keep_inputs=False,
                                        task_info={'chunk_id': cid, 'chunk_size': len(chunk),
                                                   'level': level, 'model_id': mid})

        self.logger.info('Finished submitting molecules for inference')

    @event_responder(event_name='start_inference')
    def selector(self):
        """Re-prioritize the machine learning tasks"""

        # Hold until the inference tasks are assembled
        self.logger.info('Waiting for inference tasks to be readied')
        self.inference_ready.wait()

        #  Make arrays that will hold the output results from each run
        n_tasks = 0
        y_preds = {}
        for level, chunks in self.inference_results.items():
            model_spec = self.search_spec.get_models(level)
            n_models = len(model_spec.model_paths)
            y_preds[level] = [
                np.tile(chunk, (n_models, 1)).T for chunk in chunks
            ]
            n_tasks_level = len(y_preds[level]) * n_models
            n_tasks += n_tasks_level
            self.logger.info(f'Expecting {n_tasks_level} for {level}')

        # Collect the inference runs
        for i in range(n_tasks):
            # Wait for a result
            result = self.queues.get_result(topic='infer')
            self.logger.info(f'Received inference task {i + 1}/{n_tasks}')

            # Save the inference information to disk
            with open(self.output_dir.joinpath('inference-results.json'), 'a') as fp:
                print(result.json(exclude={'value'}), file=fp)

            # Raise an error if this task failed
            if not result.success:
                raise ValueError(
                    f'Inference failed: {result.failure_info.exception}. Check the logs for further details')

            # Store the outputs
            level = result.task_info.get('level')
            chunk_id = result.task_info.get('chunk_id')
            model_id = result.task_info.get('model_id')
            y_preds[level][chunk_id][:, model_id] += np.squeeze(result.value)
            self.logger.info(f'Processed inference task {i + 1}/{n_tasks}. '
                             f'Level: {level}. Model: {model_id}. Chunk: {chunk_id}')

        # Compute the mean and std for predictions form each level
        results = []
        for level, y_pred in y_preds.items():
            y_pred = np.concatenate(y_pred, axis=0)
            mean = y_pred.mean(axis=1)
            std = y_pred.std(axis=1) * self.search_spec.get_models(level).calibration
            results.append(pd.DataFrame({
                'inchi': self.inference_mols[level],
                'level': [level] * len(mean),
                'mean': mean,
                'std': std
            }))
        results = pd.concat(results, ignore_index=True)
        self.logger.info(f'Collected a total of {len(results)} predictions')
        self._select_molecules(results)
        
        # Save the results
        results.head(self.n_to_evaluate * 4).to_csv(self.output_dir / f'task-queue-{self.inference_batch}.csv', index=False)

        # Mark that inference is complete
        self.inference_batch += 1

        # Mark that the task list has been updated
        self.update_in_progress.clear()
        self.update_complete.set()
        self.task_queue_ready.set()

    def _select_molecules(self, results: pd.DataFrame):
        """Select a list of molecules given the predictions from each model

        Adds them to the task queue

        Args:
            results: List of predictions for each molecule in self.search_space
        """

        # Rank compounds according to the upper confidence bound
        results['ucb'] = results['mean'] + self.beta * results['std']
        
        # Promote the top N of each level and 
        #  and N random ones from each level
        results['score'] = -results['ucb']
        best_score = results['score'].min()
        for gid, group in results.groupby('level'):
            results.loc[group.index[:4], 'score'] = best_score - 1
            results.loc[group.sample(2).index, 'score'] = best_score - 1
        results.sort_values('score', ascending=True, inplace=True)
            
        # Sort such that these promoted are at the top

        # Clear out the current task queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except Empty:
                break
        self.logger.info(f'Cleared out the current task queue')

        # Push the top tasks to the list
        for rid, row in results.iterrows():
            self.task_queue.put(
                _PriorityEntry(row['score'], row['inchi'], row.to_dict())
            )
            if self.task_queue.qsize() >= self.n_to_evaluate * 4:
                break
        self.logger.info('Updated task list')
        
        # Clear out the old inference tasks now we're done
        self.inference_mols.clear()  # InChI strings of molecules. Key is the level
        self.inference_results.clear()  # Placeholder for storing the result. Key is the level
     


if __name__ == '__main__':
    # User inputs
    parser = argparse.ArgumentParser()

    # Network configuration details
    group = parser.add_argument_group(title='Network Configuration',
                                      description='How to connect to the Redis task queues and task servers, etc')
    group.add_argument("--redishost", default="127.0.0.1", help="Address at which the redis server can be reached")
    group.add_argument("--redisport", default="6379", help="Port on which redis is available")
    group.add_argument("--mongohost", default="localhost", help="Hostname for MongoDB")
    group.add_argument("--mongoport", type=int, default=27845, help="Port on which MongoDB is available")

    # Computational infrastructure information
    group = parser.add_argument_group(title='Compute Infrastructure',
                                      description='Information about how to run the tasks')
    group.add_argument("--ml-endpoint", required=True, help='FuncX endpoint ID for model training and interface')
    group.add_argument("--qc-endpoint", required=True, help='FuncX endpoint ID for quantum chemistry')
    group.add_argument("--nodes-per-task", default=1,
                       help='Number of nodes per quantum chemistry task. Only needed for NWChem', type=int)
    group.add_argument("--num-qc-workers", required=True, type=int, help="Total number of quantum chemistry workers.")
    group.add_argument("--molecules-per-ml-task", default=10000, type=int,
                       help="Number of molecules per inference chunk")

    # Problem configuration
    group = parser.add_argument_group(title='Problem Definition',
                                      description='Defining the search space, models and optimizers-related settings')
    group.add_argument('--simulation-spec', type=str, required=True,
                       help='Path to YAML file describing the list of simulation fidelities and calibration models.')
    group.add_argument('--search-space', help='Path to molecules to be screened', required=True)
    group.add_argument('--search-space-name', help='Short name to use for the molecule search space', required=True)
    group.add_argument("--search-size", default=1000, type=int,
                       help="Number of new molecules to evaluate during this search")
    group.add_argument('--retrain-frequency', default=8, type=int,
                       help="Number of completed computations that will trigger a retraining")
    group.add_argument("--beta", default=1,
                       help="Degree of exploration for active learning."
                            " This is the beta from the UCB acquistion function",
                       type=float)
    group.add_argument("--pause-during-update", action='store_true',
                       help='Whether to stop running simulations while updating task list')

    # Parameters related to model retraining
    group = parser.add_argument_group(title='Model Training', description='Settings related to model retraining')
    group.add_argument('--retrain-from-scratch', action='store_true',
                       help='Whether to re-initialize weights before training')
    group.add_argument("--learning-rate", default=1e-3, help="Learning rate for re-training the models", type=float)
    group.add_argument('--num-epochs', default=512, type=int, help='Maximum number of epochs for the model training')

    # Parameters related to ProxyStore
    group = parser.add_argument_group(title='ProxyStore', description='Settings related to ProxyStore')
    group.add_argument('--simulate-ps-backend', default=None, choices=[None, 'redis', 'file', 'globus'],
                       help='ProxyStore backend to use with "simulate" topic')
    group.add_argument('--ml-ps-backend', default=None, choices=[None, 'redis', 'file', 'globus'],
                       help='ProxyStore backend to use with "infer" and "train" topics')
    group.add_argument('--ps-threshold', default=5000, type=int,
                       help='Min size in bytes for transferring objects via ProxyStore')
    group.add_argument('--ps-file-dir', default='proxy-store-scratch',
                       help='Filesystem directory to use with the ProxyStore file backend')
    group.add_argument('--simulate-ps-globus-config', default=None,
                       help='Globus Endpoint config file to use with the ProxyStore Globus backend')
    group.add_argument('--ml-ps-globus-config', default=None,
                       help='Globus Endpoint config file to use with the ProxyStore Globus backend')

    # Parse the arguments
    args = parser.parse_args()
    run_params = args.__dict__

    # Connect to MongoDB
    database = MoleculePropertyDB.from_connection_info(hostname=args.mongohost, port=args.mongoport)

    # Get the target level of accuracy
    with open(args.simulation_spec) as fp:
        simulation_spec = MultiFidelitySearchSpecification.parse_obj(yaml.safe_load(fp))

    # Create an output directory with the time and run parameters
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    out_dir = os.path.join('runs',
                           f'{simulation_spec.target_property}-N{args.num_qc_workers}-n{args.nodes_per_task}-'
                           f'{params_hash}-{start_time.strftime("%d%b%y-%H%M%S")}')
    os.makedirs(out_dir, exist_ok=False)

    # Save the run parameters to disk
    with open(os.path.join(out_dir, 'run_params.json'), 'w') as fp:
        json.dump(run_params, fp, indent=2)
    with open(os.path.join(out_dir, 'environment.json'), 'w') as fp:
        json.dump(dict(os.environ), fp, indent=2)
    with open(os.path.join(out_dir, 'simulation-spec.json'), 'w') as fp:
        print(simulation_spec.json(indent=2), file=fp)

    # Save the platform information to disk
    host_info = get_platform_info()
    with open(os.path.join(out_dir, 'host_info.json'), 'w') as fp:
        json.dump(host_info, fp, indent=2)

    # Set up the logging
    handlers = [logging.FileHandler(os.path.join(out_dir, 'runtime.log'), mode='w'),
                logging.StreamHandler(sys.stdout)]
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, handlers=handlers)
    logging.info(f'Run directory: {out_dir}')

    # Make the PS files directory inside the run directory
    ps_file_dir = os.path.abspath(os.path.join(out_dir, args.ps_file_dir))
    os.makedirs(ps_file_dir, exist_ok=False)
    logging.info(f'Scratch directory for ProxyStore files: {ps_file_dir}')

    # Init ProxyStore backends for simulation and ML tasks
    ps_names = {}
    for name, backend, globus_cfg in zip(['ml', 'sim'], [args.ml_ps_backend, args.simulate_ps_backend],
                                         [args.ml_ps_globus_config, args.simulate_ps_globus_config]):
        if backend == 'redis':
            ps.store.init_store(ps.store.STORES.REDIS, name=name, hostname=args.redishost, port=args.redisport)
        elif backend == 'file':
            ps.store.init_store(ps.store.STORES.FILE, name=name, store_dir=ps_file_dir)
        elif backend == 'globus':
            endpoints = ps.store.globus.GlobusEndpoints.from_json(globus_cfg)
            ps.store.init_store(ps.store.STORES.GLOBUS, name=name, endpoints=endpoints, timeout=3600)
        else:
            raise ValueError(f'Unsupported backend: {backend}')
    ps_names = {'simulate': 'sim', 'infer': 'ml', 'train': 'ml'}

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
    my_evaluate_mpnn = partial(evaluate_mpnn, batch_size=512, cache=True)
    my_evaluate_mpnn = update_wrapper(my_evaluate_mpnn, evaluate_mpnn)

    my_update_mpnn = partial(update_mpnn, num_epochs=args.num_epochs, learning_rate=args.learning_rate, bootstrap=True,
                             timeout=2700)
    my_update_mpnn = update_wrapper(my_update_mpnn, update_mpnn)

    my_retrain_mpnn = partial(retrain_mpnn, num_epochs=args.num_epochs, learning_rate=args.learning_rate,
                              bootstrap=True, timeout=2700)
    my_retrain_mpnn = update_wrapper(my_retrain_mpnn, retrain_mpnn)

    my_retrain_schnet = partial(train_schnet, num_epochs=args.num_epochs, learning_rate=args.learning_rate,
                                bootstrap=True, timeout=2700, property_name='delta', device='cuda')
    my_retrain_schnet = update_wrapper(my_retrain_schnet, train_schnet)

    my_evaluate_schnet = partial(evaluate_schnet, property_name='delta', device='cuda')
    my_evaluate_schnet = update_wrapper(my_evaluate_schnet, evaluate_schnet)

    # Create the task servers
    fx_client = FuncXClient()
    task_map = dict((f, args.ml_endpoint) for f in [my_evaluate_mpnn, my_update_mpnn, my_retrain_mpnn,
                                                    my_retrain_schnet, my_evaluate_schnet])
    task_map[run_single_point] = args.qc_endpoint
    task_map[relax_structure] = args.qc_endpoint
    doer = FuncXTaskServer(task_map, fx_client, server_queues)

    # Configure the "thinker" application
    thinker = Thinker(
        queues=client_queues,
        database=database,
        search_space=args.search_space,
        search_space_name=args.search_space_name,
        n_to_evaluate=args.search_size,
        n_complete_before_retrain=args.retrain_frequency,
        retrain_from_initial=args.retrain_from_scratch,
        inference_chunk_size=args.molecules_per_ml_task,
        num_qc_workers=args.num_qc_workers,
        search_spec=simulation_spec,
        output_dir=out_dir,
        beta=args.beta,
        pause_during_update=args.pause_during_update,
        ps_names=ps_names
    )
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
    for ps_backend in {'ml', 'sim'}:
        if ps_backend is not None:
            ps.store.get_store(ps_backend).cleanup()
