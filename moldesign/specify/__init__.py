"""Tools for specifying a molecular design run"""
from glob import glob
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Iterable, Union, Tuple, Any

import numpy as np
import tensorflow as tf
import torch
from pydantic import BaseModel, Field

from moldesign.score.mpnn import MPNNMessage, custom_objects
from moldesign.score.schnet import TorchMessage
from moldesign.store.models import MoleculeData, OxidationState
from moldesign.store.mongo import MoleculePropertyDB
from moldesign.store.recipes import apply_recipes, get_recipe_by_name
from moldesign.utils.conversions import convert_string_to_dict


class ModelType(str, Enum):
    MPNN = "mpnn"
    SCHNET = "schnet"


class ModelEnsemble(BaseModel):
    """Collection of models used to predict a property given the value at a lower fidelity

    Used as part of a larger specification of a set of models used to inform decisions in """

    # Basic information about the model
    base_fidelity: Optional[str] = Field(..., help='Level of fidelity used as input. Name of a RedoxRecipe')
    model_type: ModelType = Field(..., help='Type of the model. Either mpnn or schnet')

    # List of where to find the models
    model_pattern: str = Field(..., help='Pattern to matches the location of model files.'
                                         ' (e.g., model_dir/**/best.hdf5)')
    max_models: Optional[int] = Field(None, help='Maximum number of models to use for training or inference')

    # Uncertainty calibration value
    calibration: float = Field(1, help='Factor by which to scale confidence intervals')

    # Class variables that are cached
    model_paths_: List[Path] = ()

    @property
    def model_paths(self):
        """Get the paths to the models"""
        if len(self.model_paths_) == 0:
            self.model_paths_ = sorted(map(Path, glob(self.model_pattern, recursive=True)))
            if self.max_models is not None:
                self.model_paths_ = self.model_paths_[:self.max_models]
        return self.model_paths_

    def load_all_model_messages(self) -> Iterable[Union[MPNNMessage, TorchMessage]]:
        """Load the models to disk and prepare them in a format ready to send to workers

        Yields:
            Model in serializable format
        """

        for path in self.model_paths:
            yield self._create_message(path)

    def _create_message(self, path: Path) -> Union[MPNNMessage, TorchMessage]:
        """Create a message for a model at a certain path

        Args:
            path: Path to the model file
        Returns:
            Model in serializable format
        """
        if self.model_type == ModelType.MPNN:
            model = tf.keras.models.load_model(path, custom_objects=custom_objects)
            return MPNNMessage(model)
        elif self.model_type == ModelType.SCHNET:
            model = torch.load(path, map_location=torch.device('cpu'))
            return TorchMessage(model)
        else:
            raise NotImplementedError(f'Loading not implemented for {self.model_type}')

    def load_model_message(self, index: int) -> Union[MPNNMessage, TorchMessage]:
        """Load a model from disk and prepare in format ready to send to workers

        Args:
            index: Index of the model
        """
        return self._create_message(self.model_paths[index])

    def update_model(self, index: int, message: Union[List, TorchMessage]):
        """Update the state of one of the models

        Args:
            index: Index of the model
            message: Update information. New weights for "mpnn" and new model for "schnet"
        """

        path = self.model_paths_[index]
        if self.model_type == ModelType.MPNN:
            model = tf.keras.models.load_model(path, custom_objects=custom_objects)
            model.set_weights(message)
            tf.keras.models.save_model(model, path)
        elif self.model_type == ModelType.SCHNET:
            torch.save(message.get_model(map_location='cpu'), path)
        else:
            raise NotImplementedError(f'Loading not implemented for {self.model_type}')


class MultiFidelitySearchSpecification(BaseModel):
    """Definition for a multi-fidelity search.

    Includes the target level fidelity as well as all of the easier-to-compute levels at earlier steps.
    We define the intermediate steps for computing a property with high-fidelity by providing
    the list of models used to calibrate the results of low-fidelity calculations.
    """

    # Defining the simulation steps
    oxidation_state: OxidationState = Field(..., help='Oxidation state we are assessing')
    target_level: str = Field(...,help='Name of the recipe used as the highest level of accuracy')

    # Model definitions
    model_levels: List[ModelEnsemble] = Field(..., help='Models to calibrate at different levels of fidelity.')
    base_model: ModelEnsemble = Field(..., help='Models used when no information is available about a molecule')

    @property
    def levels(self):
        """List of the fidelity levels to consider"""
        return [level.base_fidelity for level in self.model_levels] + [self.target_level]

    @property
    def target_property(self):
        if self.oxidation_state == OxidationState.OXIDIZED:
            return f'oxidation_potential.{self.target_level}'
        else:
            return f'reduction_potential.{self.target_level}'

    def get_base_training_set(self, database: MoleculePropertyDB) -> Dict[str, float]:
        """Get the training set for the base model

        Args:
            database: Connection to a collection of molecular properties
        Returns:
            Training set used to train "molecule structure" -> "property" models
        """

        results = database.get_training_set(['identifier.smiles'], [self.target_property])
        return dict(zip(results['identifier.smiles'], results[self.target_property]))

    def get_calibration_training_set(self, level: int, database: MoleculePropertyDB) -> Dict[str, float]:
        """Get the training set for a certain level of fidelity

        Args:
            level: Index of the desired level of fidelity
            database: Connection to a collection of molecular properties
        Returns:
            Training set useful for that calibration model
        """

        # Get the recipe level of fidelity used as the base
        recipe = get_recipe_by_name(self.model_levels[level].base_fidelity)

        # Define the name of the input description of the molecule
        model_type = self.model_levels[level].model_type
        if model_type == ModelType.SCHNET:
            #  Use the geometry at the base level of fidelity, and select the charged geometry only if available
            xyz = f'data.{recipe.geometry_level}.{self.oxidation_state if recipe.adiabatic else "neutral"}.xyz'
        else:
            #  Use the SMILES string
            xyz = 'identifier.smiles'

        # Get the low-res level of fidelity
        low_res = 'oxidation_potential' if self.oxidation_state == OxidationState.OXIDIZED else 'reduction_potential'
        low_res += '.' + recipe.name

        # Query the database to be the output
        results = database.get_training_set([xyz, low_res], [self.target_property])

        # Compute the delta between base and target
        delta = np.subtract(results[self.target_property], results[low_res])

        # Return that as the training set
        return dict(zip(results[xyz], delta))

    def get_inference_inputs(self, record: MoleculeData) -> Tuple[str, Any, float]:
        """Determine which model to use for inference and the inputs needed for that model

        Args:
            record: Molecule to evaluate
        Returns:
            - Name of the model that should be run
            - Inputs to the machine learning model
            - Value to be calibrated
        """

        # Determine which model to run
        current_step = self.get_current_step(record)

        # Get the model spec for that level and the input value
        if current_step == 'base':
            model_spec = self.base_model
            init_value = 0
        else:
            model_spec = self.get_models(current_step)
            init_value = record.oxidation_potential[current_step] if self.oxidation_state == OxidationState.OXIDIZED \
                else record.reduction_potential[current_step]

        # Get the inputs
        model_type = model_spec.model_type
        if model_type == ModelType.SCHNET:
            recipe = get_recipe_by_name(current_step)
            # Use the geometry at the base level of fidelity, and select the charged geometry only if available
            input_val = record.data[recipe.geometry_level][self.oxidation_state if recipe.adiabatic else "neutral"].xyz
        elif model_type == ModelType.MPNN:
            # Use a dictionary
            input_val = convert_string_to_dict(record.identifier['inchi'])
        else:
            raise NotImplementedError(f'No support for {model_type} yet')

        return current_step, input_val, init_value

    def get_models(self, step_name) -> ModelEnsemble:
        """Get the model specification for a certain level

        Args:
            step_name: Name of the step
        Returns:
            Associated specification
        """

        if step_name == 'base':
            return self.base_model

        for model_spec in self.model_levels:
            if model_spec.base_fidelity == step_name:
                return model_spec
        raise ValueError(f'Spec not found for {step_name}')

    def get_next_step(self, record: MoleculeData) -> Optional[str]:
        """Get the next fidelity level for a certain molecule given what we know about it

        Args:
            record: Molecule to be evaluated
        Returns:
            The name of the next level of fidelity needed for this computation. ``None`` if all have been completed
        """

        # Make sure all of our evaluations are up-to-date
        apply_recipes(record)

        # Get the appropriate property we are looking through
        data = record.reduction_potential if self.oxidation_state == OxidationState.REDUCED \
            else record.oxidation_potential

        # If the highest level is found, return None as we're done
        if self.levels[-1] in data:
            return None

        # Get the first level to be found in the molecule
        current_level = None
        for level in self.levels[::-1]:
            if level in data:
                current_level = level
                break

        # If no level has been completed, return the first level
        if current_level is None:
            return self.levels[0]

        # Otherwise, return the next level in the chain
        return self.levels[self.levels.index(current_level) + 1]

    def get_current_step(self, record: MoleculeData) -> str:
        """Get the current level of fidelity for a certain molecule

        Args:
            record: Molecule to be evaluated
        Returns:
            The name of the highest-level achieved so far. "base" if the molecule has yet to be assessed
        """

        # Make sure all of our evaluations are up-to-date
        apply_recipes(record)

        # Get the appropriate property we are looking through
        data = record.reduction_potential if self.oxidation_state == OxidationState.REDUCED \
            else record.oxidation_potential

        # Get the current level
        for level in self.levels[::-1]:
            if level in data:
                return level
        return 'base'
