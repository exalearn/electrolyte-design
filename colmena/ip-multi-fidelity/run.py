from threading import Event, Barrier
from typing import List, Dict, Union, Tuple, Optional
from functools import partial, update_wrapper
from urllib import parse
from pathlib import Path
from datetime import datetime
from queue import Empty, PriorityQueue
from dataclasses import dataclass, field
from collections import defaultdict
import argparse
import logging
import hashlib
import json
import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from scipy.stats.distributions import norm
from colmena.task_server import ParslTaskServer
from colmena.models import Result
from colmena.redis.queue import ClientQueues, make_queue_pairs
from colmena.thinker import BaseThinker, result_processor, event_responder, task_submitter
from colmena.thinker.resources import ResourceCounter

from config import theta_nwchem_config
from moldesign.score.mpnn import evaluate_mpnn, update_mpnn, retrain_mpnn, custom_objects
from moldesign.score.schnet import train_schnet, evaluate_schnet
from moldesign.store.models import MoleculeData, OxidationState, UnmatchedGeometry, RedoxEnergyRecipe
from moldesign.store.mongo import MoleculePropertyDB
from moldesign.store.recipes import apply_recipes, get_recipe_by_name
from moldesign.utils.chemistry import get_baseline_charge
from moldesign.utils import get_platform_info
from sim import compute_adiabatic, compute_vertical, compute_single_point

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


class Thinker(BaseThinker):
    """ML-enhanced optimization loop for molecular design"""

    def __init__(self, queues: ClientQueues,
                 num_workers: int,
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
            num_workers: Number of nodes available for computations
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
        super().__init__(queues, ResourceCounter(num_workers, ['training', 'inference', 'simulation']), daemon=True)

        # Configuration for the run
        self.inference_chunk_size = inference_chunk_size
        self.n_complete_before_reorder = n_complete_before_reorder
        self.n_complete_before_retrain = n_complete_before_retrain
        self.retrain_from_initial = retrain_from_initial
        self.models = models.copy()
        self.n_models = sum(map(len, models.values()))
        self.calibration_factor = calibration_factor.copy()
        self.output_dir = Path(output_dir)
        self.beta = beta
        self.nodes_per_qc = nodes_per_qc
        self.random_seed = random_seed
        self.target_recipe = target_recipe
        self.oxidize = oxidize
        self.oxidation_state = OxidationState.OXIDIZED if oxidize else OxidationState.REDUCED
        self.output_property =  ('oxidation' if oxidize else 'reduction') + f"_potential.{target_recipe.name}"
        self.solvent = target_recipe.solvent
        self.target_range = target_range

        # Store link to the database
        self.database = database

        # Get the target database size
        already_done = self.database.get_training_set(['identifier.inchi'],
                                                      [self.output_property])['identifier.inchi']
        self.logger.info(f'Computations of {self.output_property} already complete for {len(already_done)} molecules')
        self.n_to_evaluate = n_to_evaluate
        self.n_evaluated = 0
        self.until_retrain = self.n_complete_before_retrain
        self.until_reevaluate = self.n_complete_before_reorder

        # Mark where we have worked before
        self.already_ran = defaultdict(set)

        # Prepare search space for low and high fidelity computation
        self.search_space = set(search_space)
        self.yet_unevaluated = self.search_space.copy()  # Start off with all of the molecules
        self.logger.info(f'Loaded a search space of {len(self.yet_unevaluated)} molecules')

        self.yet_unevaluated.difference_update(already_done)
        self.logger.info(f'A total of {len(self.yet_unevaluated)} have yet to be fully-evaluated')
        output_tag = 'oxidation_potential' if self.oxidize else 'reduction_potential'
        for low in [f'{output_tag}.smb-vacuum-vertical', f'{output_tag}.smb-vacuum-no-zpe']:
            low_fidelity_ready = self.database.get_eligible_molecules([low], [self.output_property])['identifier.inchi']
            self.logger.info(f'There are {len(low_fidelity_ready)} molecules with {low} available.'
                             f' Some may be outside the search space')
            self.yet_unevaluated.difference_update(low_fidelity_ready)
        self.logger.info(f'There are {len(self.yet_unevaluated)} molecules ready for the low-fidelity computation')

        # Inter-thread communication stuff
        self.start_inference = Event()  # Mark that inference should start
        self.start_training = Event()  # Mark that retraining should start
        self.task_queue = PriorityQueue()  # Holds a list of tasks for inference

        self.training_done = Barrier(2)

        self.to_reevaluate: Dict[str, List[MoleculeData]] = defaultdict(list)  # List of molecules with new data
        self.new_model = True  # Used to determine if we need to re-compute scores for all molecules
        self.inference_batch = 0
        self.inference_ready = Barrier(2)  # Wait until all agents are ready for inference, both complete
        self.inference_results = {}  # Output for inference results
        self.inference_inputs = {}  # InChI and, optionally,

        # Allocate maximum number of ML workers to inference, remainder to simulation
        self.max_ml_workers = 1
        self.rec.reallocate(None, 'inference', num_workers)

        # Trigger the inference to start
        self.start_inference.set()

    @task_submitter(task_type='simulation', n_slots='nodes_per_qc')
    def launch_qc(self):
        # Get the next task
        entry: _PriorityEntry = self.task_queue.get()
        sim_info = entry.item

        # Determine which level we are running
        method = sim_info['method']
        inchi = sim_info['inchi']

        # Get some basic information on the molecule
        record = self.database.get_molecule_record(inchi=inchi)
        neutral_charge = get_baseline_charge(inchi)
        redox_charge = neutral_charge + (1 if self.oxidize else -1)

        self.logger.info(f'Running {method} computation(s) for {inchi}')
        self.already_ran[method].add(inchi)
        if method == 'compute_vertical':
            self.yet_unevaluated.remove(inchi)  # We've started to gather data for it

            # Check if we need to run the neutral relaxation
            if self.target_recipe.geometry_level in record.data and \
                    OxidationState.NEUTRAL in record.data[self.target_recipe.geometry_level]:
                # If already have the neutral, just submit the vertical
                neutral_xyz = record.data[self.target_recipe.geometry_level][OxidationState.NEUTRAL].xyz
                self.queues.send_inputs(neutral_xyz, redox_charge, None, self.target_recipe.geometry_level,
                                        method='compute_single_point', task_info=sim_info,
                                        topic='simulate', keep_inputs=True)
            else:
                self.queues.send_inputs(inchi, self.oxidize, task_info=sim_info,
                                        method='compute_vertical', keep_inputs=True,
                                        topic='simulate')
        elif method == 'compute_adiabatic':
            xyz = sim_info['xyz']
            self.queues.send_inputs(xyz, neutral_charge, self.oxidize, task_info=sim_info,
                                    method='compute_adiabatic', keep_inputs=True,
                                    topic='simulate')
        elif method == 'compute_single_point':
            # Determine which computations we need to run, defined by the geometry and the charge
            to_run: List[Tuple[str, int, Optional[str]]] = []
            geom_level_data = record.data[self.target_recipe.geometry_level]
            for (ox_state, charge) in zip((OxidationState.NEUTRAL, self.oxidation_state),
                                       (neutral_charge, redox_charge)):
                # See if we've done the computation in vacuum
                if self.target_recipe.energy_level not in geom_level_data[ox_state].total_energy[ox_state]:
                    to_run.append((geom_level_data[ox_state].xyz, charge, None))

                # Check if we need the computation in solvent
                if self.solvent is not None and \
                        (ox_state not in geom_level_data[ox_state].total_energy_in_solvent or
                         self.solvent not in geom_level_data[ox_state].total_energy_in_solvent[ox_state] or 
                         self.target_recipe.energy_level not in geom_level_data[ox_state].total_energy_in_solvent[ox_state][self.solvent]):
                    to_run.append((geom_level_data[ox_state].xyz, charge, self.solvent))
            self.logger.info(f'Submitting {len(to_run)} single point computations to finish high-fidelity')

            # Send the single points one after each other
            first = True
            for run_args in to_run:
                # We need to request more resources after the first submission
                if first:
                    first = False
                else:
                    self.rec.acquire('simulation', self.nodes_per_qc)
                self.queues.send_inputs(*run_args, self.target_recipe.energy_level, task_info=sim_info,
                                        method='compute_single_point', keep_inputs=True,
                                        topic='simulate')
        else:
            raise ValueError(f'Method "{method}" not recognized')

    @result_processor(topic='simulate')
    def record_qc(self, result: Result):
        # Get basic task information
        inchi = result.task_info['inchi']
        self.logger.info(f'{result.method} computation for {inchi} finished')

        # Release nodes for use by other processes
        self.rec.release("simulation", self.nodes_per_qc)

        # If successful, add to the database
        if result.success:
            self.n_evaluated += 1

            # Check if we are done
            if self.n_evaluated >= self.n_to_evaluate:
                self.logger.info(f'We have evaluated as many molecules as requested. exiting')
                self.done.set()

            # Write outputs to disk
            opt_records, spe_records = result.value
            with open(self.output_dir.joinpath('..', '..', 'qcfractal-records.json'), 'a') as fp:
                for r in opt_records + spe_records:
                    r.extras['inchi'] = inchi
                    print(r.json(), file=fp)

            # Store the data in a molecule data object
            data = self.database.get_molecule_record(inchi=inchi)  # Get existing information
            store_success = False
            try:
                for r in opt_records:
                    data.add_geometry(r, overwrite=True)
                for r in spe_records:
                    data.add_single_point(r)
                store_success = True
            except UnmatchedGeometry:
                self.logger.warning(f'Failed to match {inchi} geometry to an existing record.'
                                    ' Tell Logan his hashes are broken again!')
            apply_recipes(data)  # Compute the IP

            # Add ionization potentials to the task_info
            result.task_info['ips'] = data.oxidation_potential
            result.task_info['eas'] = data.reduction_potential

            # Add to database
            with open(self.output_dir.joinpath('moldata-records.json'), 'a') as fp:
                print(json.dumps([datetime.now().timestamp(), data.json()]), file=fp)
            self.database.update_molecule(data)

            # Mark if we have completed a new record of the output property
            outputs = data.oxidation_potential if self.oxidize else data.reduction_potential
            if self.target_recipe.name in outputs:  # All SPE are complete
                self.until_retrain -= 1
                self.logger.info(f'High fidelity complete. {self.until_retrain} before retraining')
            elif result.task_info['method'] != "compute_single_point" and store_success:
                self.until_reevaluate -= 1
                self.logger.info(f'Low fidelity complete. {self.until_reevaluate} before re-ordering')
                if result.method == 'compute_vertical':
                    self.to_reevaluate['adiabatic'].append(data)
                else:
                    self.to_reevaluate['normal'].append(data)

            # Check if we should re-do training or re-run inference
            if self.until_retrain <= 0 and not self.done.is_set():
                # If we have enough new
                self.logger.info('Triggering training to start')
                self.start_training.set()
            elif self.until_reevaluate <= 0 and not (self.start_training.is_set() or self.done.is_set()):
                # Restart inference if we have had enough complete computations
                self.logger.info('Triggering inference to begin again')
                self.start_inference.set()

            self.logger.info(f'Added complete calculation for {inchi} to database.')
        else:
            self.logger.info(f'Computations failed for {inchi}. Check JSON file for stacktrace')

        # Write out the result to disk
        with open(self.output_dir.joinpath('simulation-results.json'), 'a') as fp:
            print(result.json(exclude={'value'}), file=fp)

    @event_responder(event_name='start_training', reallocate_resources=True, max_slots=24,
                     gather_from='simulation', gather_to='training', disperse_to='inference',
                     slot_step=4)
    def train_models(self):
        """Train machine learning models"""
        self.logger.info('Started retraining')

        # Start by training the MPNN
        train_data = self.database.get_training_set(['identifier.smiles'], [self.output_property])
        train_data = dict(zip(train_data['identifier.smiles'], train_data[self.output_property]))
        self.logger.info(f'Gathered {len(train_data)} molecules to train the MPNN')

        mpnn_config = None
        if self.retrain_from_initial:
            # Get a copy of the model configuration
            mpnn_config = tf.keras.models.load_model(self.models['vertical'][0],
                                                     custom_objects=custom_objects).get_config()

        for mid, model in enumerate(self.models['vertical']):
            # Wait until we have nodes
            if not self.rec.acquire('training', 1, cancel_if=self.done):
                # If unsuccessful, exit because we are finished
                return

                # Make the MPNN message
            if self.retrain_from_initial:
                self.queues.send_inputs(mpnn_config, train_data, method='retrain_mpnn', topic='train',
                                        task_info={'model_id': mid, 'level': 'vertical'},
                                        keep_inputs=False,
                                        input_kwargs={'random_state': mid + self.random_seed})
            else:
                self.queues.send_inputs(model, train_data, method='update_mpnn', topic='train',
                                        task_info={'model_id': mid, 'level': 'vertical'},
                                        keep_inputs=False,
                                        input_kwargs={'random_state': mid + self.random_seed})
            self.logger.info(f'Submitted MPNN {mid} to train with {len(train_data)} entries')

        # Then train the SchNet models
        pot_name = 'oxidation_potential' if self.oxidize else 'reduction_potential'
        ox_name = OxidationState.OXIDIZED if self.oxidize else OxidationState.REDUCED
        for level, xyz, low_res in [
            ('adiabatic', 'data.small_basis.neutral.xyz', f'{pot_name}.smb-vacuum-vertical'),
            ('normal', f'data.small_basis.{ox_name}.xyz', f'{pot_name}.smb-vacuum-no-zpe')
        ]:
            train_data = self.database.get_training_set([xyz, low_res], [self.output_property])
            train_data = pd.DataFrame(train_data)
            train_data['delta'] = train_data[self.output_property] - train_data[low_res]
            train_data = dict(zip(train_data[xyz], train_data['delta']))
            self.logger.info(f'Gathered {len(train_data)} molecules to train the {level} SchNet models')

            for mid, model in enumerate(self.models[level]):
                # Wait until we have nodes
                if not self.rec.acquire('training', 1, cancel_if=self.done):
                    # If unsuccessful, exit because we are finished
                    return

                self.queues.send_inputs(model, train_data, method='train_schnet', topic='train',
                                        task_info={'model_id': mid, 'level': level},
                                        keep_inputs=False,
                                        input_kwargs={'random_state': mid + self.random_seed,
                                                      'reset_weights': self.retrain_from_initial})
                self.logger.info(f'Submitted {level} {mid} to train with {len(train_data)} entries')

        # Wait for the run to finish
        self.logger.info('Done submitting training jobs')
        self.training_done.wait()
        self.logger.info('Waiting until next training requested')

    @event_responder(event_name='start_training')
    def update_weights(self):
        """Process the results of the saved model"""
        self.logger.info('Waiting for updated weights')

        # Save the model data, as appropriate
        n_models = sum(len(v) for v in self.models.values())
        for i in range(n_models):
            # Wait for a training result to complete
            result = self.queues.get_result(topic='train')
            self.rec.release('training', 1)

            # Save results to disk
            with open(self.output_dir.joinpath('training-results.json'), 'a') as fp:
                print(result.json(exclude={'inputs', 'value'}), file=fp)

            self.logger.info(f'Received training result {i + 1}/{n_models}.'
                             f' Type: {result.method}. Success: {result.success}')
            model_id = result.task_info['model_id']
            level = result.task_info['level']
            self.logger.info(f'Training completed for {level} model {model_id}')

            if not result.success:
                self.logger.warning(f'Training failed for {result.method} {model_id}')
                continue

            # Store the result as appropriate
            if result.method == 'train_schnet':
                self.logger.info(f'SchNet {model_id} finished training.')

                # Save the new model to disk
                model_path = self.models[level][model_id]
                model_msg, history = result.value
                model = model_msg.get_model()
                torch.save(model, model_path)
                self.logger.info(f'Saved updated SchNet to {model_path}')

                # Save the history
                with open(self.output_dir.joinpath('schnet-training-history.json'), 'a') as fp:
                    print(history.to_json(), file=fp)
            else:
                self.logger.info(f'MPNN {model_id} finished training.')

                # Update weights in the h5
                weights, history = result.value
                model_path = self.models[level][model_id]
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                model.set_weights(weights)
                model.save(model_path)
                self.logger.info(f'Saved updated MPNN to {model_path}')

                # Save the history
                with open(self.output_dir.joinpath('mpnn-training-history.json'), 'a') as fp:
                    print(repr(history), file=fp)

        # Once all models are finished, set when we should train again and reset the flag
        self.until_retrain = self.n_complete_before_retrain
        self.logger.info('Done submitting training jobs')
        self.training_done.wait()
        self.start_training.clear()

        # Make inference begin
        self.new_model = True  # Makes inference tasks re-run everything
        self.start_inference.set()
        self.logger.info('Waiting until next training requested')

    @event_responder(event_name='start_inference', reallocate_resources=True, max_slots=None,
                     gather_from='simulation', gather_to='inference', disperse_to='simulation',
                     slot_step=4)
    def launch_inference(self):
        """Submit inference tasks for the yet-unlabelled samples"""
        self.logger.info('Beginning to submit inference tasks')

        # Pull the search space in a compact form
        #  MPNN: We need the inchi strings
        #  SchNet: We need the InChI, neutral geometry and low-fidelity IP
        pot_name = 'oxidation_potential' if self.oxidize else 'reduction_potential'
        ox_name = OxidationState.OXIDIZED if self.oxidize else OxidationState.REDUCED
        if self.new_model:  # Run all molecules that are eligible for either low or high-fidelity
            self.logger.info('Re-running all molecules')
            batch_search_space = {
                'vertical': list(self.yet_unevaluated),  # Get a snapshot, in case it changes
                'adiabatic': [x for x in self.database.get_eligible_molecule_records(
                    ['identifier.inchi', 'data.small_basis.neutral', f'{pot_name}.smb-vacuum-vertical'],
                    [self.output_property]) if x.identifier['inchi'] in self.search_space],
                'normal': [x for x in self.database.get_eligible_molecule_records(
                    ['identifier.inchi', f'data.small_basis.{ox_name}', f'{pot_name}.smb-vacuum-no-zpe'],
                    [self.output_property]) if x.identifier['inchi'] in self.search_space]
            }
        else:  # Only run molecules where a simulation finished since the last re-ordering
            self.logger.info('Only running molecules with new simulation data')
            batch_search_space = self.to_reevaluate.copy()
            batch_search_space['vertical'] = []
        self.to_reevaluate.clear()  # We always re-evaluate these points

        # Prepare storage for the inference recorder / task selector thread
        inference_chunks = {}  # Holds the inference for each type of model
        self.inference_results.clear()  # Holds the storage arrays for the outputs
        self.inference_inputs.clear()  # Holds the inputs to the simulations

        # Detail the inputs and result storage for the MPNN
        if len(batch_search_space['vertical']) > 0:
            self.inference_inputs['vertical'] = [{'inchi': x, 'method': 'compute_vertical'}
                                                 for x in batch_search_space['vertical']]
            inference_chunks['vertical'] = np.array_split(batch_search_space['vertical'],
                                                          len(batch_search_space['vertical']) //
                                                          self.inference_chunk_size + 1)
            self.inference_results['vertical'] = [
                np.zeros((len(x), len(self.models['vertical'])), dtype=np.float32) for x in inference_chunks['vertical']
            ]
            self.logger.info(f'Submitting {len(self.inference_inputs["vertical"])} molecules to evaluate with MPNN.')

        # Detail the inputs and result storage for the delta learning models
        for tag, xyz_charge, low_res, method in [
            ('adiabatic', OxidationState.NEUTRAL, 'smb-vacuum-vertical', 'compute_adiabatic'),
            ('normal', OxidationState.OXIDIZED, 'smb-vacuum-no-zpe', 'compute_single_point')
        ]:
            if len(batch_search_space[tag]) > 0:
                # Unpack the XYZ and low-resolution IP

                xyzs = [x.data['small_basis'][xyz_charge].xyz for x in batch_search_space[tag]]
                low_res_vals = [x.oxidation_potential[low_res] if self.oxidize else x.reduction_potential[low_res]
                                for x in batch_search_space[tag]]

                inference_chunks[tag] = np.array_split(
                    xyzs, len(batch_search_space[tag]) // self.inference_chunk_size + 1
                )
                self.inference_inputs[tag] = [
                    {'inchi': x.identifier['inchi'],
                     'xyz': xyz,
                     'low_fidelity': lr,
                     'method': method}
                    for x, xyz, lr in zip(batch_search_space[tag], xyzs, low_res_vals)
                ]
                init_schnet = np.tile(
                    np.array(low_res_vals, dtype=np.float32)[:, None],
                    (1, len(self.models[tag]))
                )  # Tile to create the same initial value for each model
                self.inference_results[tag] = np.array_split(init_schnet, len(inference_chunks[tag]))
                self.logger.info(f'Submitting {len(xyzs)} molecules to evaluate with SchNet for {tag}.')

        # Mark that we are ready to start inference
        self.inference_ready.wait()
        self.logger.info(f'Initialized arrays in which to store inference results')

        # Submit the SMILES to run inference
        for tag, chunks in inference_chunks.items():
            model_type = 'mpnn' if tag == 'vertical' else 'schnet'
            for mid, model in enumerate(self.models[tag]):
                for cid, chunk in enumerate(chunks):
                    self.rec.acquire('inference', 1)
                    self.queues.send_inputs([model], chunk.tolist(),
                                            topic='infer', method=f'evaluate_{model_type}',
                                            keep_inputs=False,
                                            task_info={'chunk_id': cid, 'chunk_size': len(chunk),
                                                       'tag': tag, 'model_id': mid})
                    self.logger.info(f'Submitted chunk {cid + 1}/{len(chunks)}'
                                     f' for {tag} {mid + 1}/{len(self.models[tag])}')
            self.logger.info(f'Finished submitting {tag} for inference')

        # Wait for recorder to finish
        self.logger.info(f'Waiting for inference to complete before exiting')
        self.inference_ready.wait()

    @event_responder(event_name='start_inference')
    def record_inference(self):
        """Re-prioritize the machine learning tasks"""
        # Wait until the launch thread counts how many interface tasks to submit
        self.inference_ready.wait()

        # Collect the inference runs
        n_inference_tasks = sum(len(v) * v[0].shape[1] for v in self.inference_results.values())
        self.logger.info(f'Waiting to receive {n_inference_tasks} inference results')
        for i in range(n_inference_tasks):
            # Wait for a result
            result = self.queues.get_result(topic='infer')
            self.rec.release('inference', 1)
            self.logger.info(f'Received inference task {i + 1}/{n_inference_tasks}')

            # Save the inference information to disk
            with open(self.output_dir.joinpath('inference-records.json'), 'a') as fp:
                print(result.json(exclude={'value'}), file=fp)

            # Determine the data
            if not result.success:
                raise ValueError('Result failed! Check the JSON')

            # Store the outputs
            chunk_id = result.task_info.get('chunk_id')
            model_id = result.task_info.get('model_id')
            tag = result.task_info.get('tag')
            y_pred = self.inference_results[tag]
            y_pred[chunk_id][:, model_id] += np.squeeze(result.value)
        self.logger.info('All inference tasks are complete')

        # Compute the mean and std for each prediction, scale by the calibration factor
        y_mean = {}
        y_std = {}
        for tag, y_pred in self.inference_results.items():
            y_pred = np.concatenate(y_pred, axis=0)
            y_mean[tag] = y_pred.mean(axis=1)
            y_std[tag] = y_pred.std(axis=1) * self.calibration_factor[tag]

        # Combine both sources into the same list
        tags = list(y_mean.keys())
        y_mean = np.concatenate([y_mean[t] for t in tags], axis=0)
        y_std = np.concatenate([y_std[t] for t in tags], axis=0)
        inputs = np.concatenate([self.inference_inputs[t] for t in tags], axis=0)
        self._select_molecules(y_mean, y_std, inputs, reset_queue=self.new_model)  # Only reset if model is "new"

        # Mark that inference is complete
        self.new_model = False  # We have now evaluated molecules with these models
        self.inference_ready.wait()
        self.until_reevaluate = self.n_complete_before_reorder
        self.start_inference.clear()
        self.inference_batch += 1

    def _select_molecules(self, y_mean, y_std, inputs, reset_queue):
        """Select a list of molecules given the predictions from each model

        Adds them to the task queue

        Args:
            y_mean: Mean of the predictions
            y_std: Standard deviation, scaled to calibrate the uncertainties
            inputs: Inputs for the simulation
            reset_queue: Whether to reset the queue before adding new entries
        """

        # Tell the user what you plan on doing
        if reset_queue:
            self.logger.info('Resetting the task queue.')
        else:
            self.logger.info(f'Adding {len(inputs)} new entries to priority queue')

        # Clear out the current queue. Prevents new simulations from starting while we pick taxes
        while reset_queue and not self.task_queue.empty():
            try:
                self.task_queue.get(False)
            except Empty:
                continue

        # Rank compounds according to the upper confidence bound
        dists = norm(loc=y_mean, scale=y_std)
        prob = dists.cdf(self.target_range[1]) - dists.cdf(self.target_range[0])
        sort_ids = list(np.argsort(prob))  # Sorted worst to best (ascending)

        # Make a temporary copy as a List, which means we can easily judge its size
        task_queue = []
        while len(task_queue) < self.n_to_evaluate * 2:
            # Special case: exit if no more left to choose
            if len(sort_ids) == 0:
                break

            # Pick a molecule
            mol_id = sort_ids.pop()  # Pops from the best side
            mol_data = inputs[mol_id]

            # Store the inference results
            mol_data['mean'] = float(y_mean[mol_id])  # Converts from np.float32, which is not JSON serializable
            mol_data['std'] = float(y_std[mol_id])
            mol_data['prob'] = float(prob[mol_id])

            # Add it to list if hasn't already been ran
            #  There are cases where a computation could have been run between when inference started and now
            inchi = mol_data['inchi']
            method = mol_data['method']
            if inchi in self.already_ran[method]:
                continue
            # Note: converting to float b/c np.float32 is not JSON serializable
            task_queue.append(mol_data)

        # Put new tasks in the queue
        self.logger.info(f'Pushing {len(task_queue)} jobs to the task queue')
        for task_info in task_queue:
            entry = _PriorityEntry(score=-task_info['prob'], item=task_info)
            self.task_queue.put(entry)  # Negative so largest UCB is first
        self.logger.info(f'Updated task list. Current size: {self.task_queue.qsize()}')


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

    # Parse the arguments
    args = parser.parse_args()
    run_params = args.__dict__

    # Define the compute setting for the system (only relevant for NWChem)
    nnodes = int(os.environ.get("COBALT_JOBSIZE", "1"))
    run_params["nnodes"] = nnodes
    run_params["qc_workers"] = nnodes / args.nodes_per_task

    # Get the desired computational settings
    recipe = get_recipe_by_name(args.target_level)

    # Connect to MongoDB
    mongo_url = parse.urlparse(args.mongo_url)
    mongo = MoleculePropertyDB.from_connection_info(mongo_url.hostname, mongo_url.port)

    # Load in the search space
    full_search = pd.read_csv(args.search_space, delim_whitespace=True)
    search_space = full_search['inchi'].values

    # Create an output directory with the time and run parameters
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    out_dir = Path('runs').joinpath(f'{args.target_level}-{start_time.strftime("%d%b%y-%H%M%S")}-{params_hash}')
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
    config = theta_nwchem_config(os.path.join(out_dir, 'run-info'), nodes_per_nwchem=args.nodes_per_task,
                                 total_nodes=nnodes)

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


    my_evaluate_mpnn = _fix_arguments(evaluate_mpnn, batch_size=args.batch_size, cache=True, n_jobs=8)
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
    my_compute_adiabatic = _fix_arguments(compute_adiabatic, spec_name=recipe.geometry_level, n_nodes=args.nodes_per_task)
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
    thinker = Thinker(client_queues,
                      nnodes,
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
