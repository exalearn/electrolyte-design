"""Tools for specifying a molecular design run"""
from enum import Enum
from typing import List, Tuple, Optional, Iterable

from pydantic import BaseModel, Field

from moldesign.simulate.functions import generate_inchi_and_xyz
from moldesign.store.models import MoleculeData, RedoxEnergyRecipe, AccuracyLevel, OxidationState
from moldesign.store.mongo import MoleculePropertyDB
from moldesign.store.recipes import get_recipe_by_name
from moldesign.utils.chemistry import get_baseline_charge


class ModelType(str, Enum):
    MPNN = "mpnn"
    SCHNET = "schnet"


class FidelityLevel(BaseModel):
    """Defines a step in a multi-fidelity design pipeline

    For each step we need to know:
    1. The target level of accuracy
    2. What calculations are necessary to achieve that level of accuracy
    3. Once that level is achieved for a molecule,
    how to predict the redox potential at the highest level of fidelity in the pipeline

    All information needed to define a level of fidelity is defined in the name of the recipe.

    We can determine which computations are needed to run by comparing the required information for the recipe
    to what is available in a particular record, and the recipe from the previous step in the pipeline.
    The previous step in the pipeline defines which points to start from.

    We define how to use this level of fidelity to predict the highest level in the pipeline by indicating the
    model type, whether that model should be trained using delta learning, and a pattern to match the list of models
    to be used.
    """

    # Information defining the quantum chemistry computations
    recipe: str = Field(..., help='Name of the recipe')

    # Information defining how we predict the outcome given
    model_type: ModelType = Field(..., help='Type of the model.')
    model_path: str = Field(..., help='Pattern that matches the location of model files. '
                                      '(e.g., model_dir/**/best.hdf5)')
    max_models: Optional[int] = Field(None, help='Maximum number of models to use for training or inference')

    def get_required_calculations(self, record: MoleculeData, oxidation_state: OxidationState,
                                  previous_level: Optional['FidelityLevel'] = None) \
            -> List[Tuple[AccuracyLevel, str, int, Optional[str], bool]]:
        """List the required computations to complete a level given the current information about a molecule

        If this method returns relaxation calculations, then there may be more yet to perform after those complete
        before one can evaluate the redox potential at the desired level of accuracy.
        Calculations that use those input geometries may be needed

        Args:
            record: Information available about the molecule
            oxidation_state: Oxidation state for the redox computation
            previous_level: Previous level of accuracy, used to determine a starting point for relaxations
        Returns:
            List of required computations as tuples of
                (level of accuracy,
                 input XYZ structure,
                 charge state,
                 solvent,
                 whether to relax)
            All computations can be performed in parallel
        """

        # Required computations
        required = []

        # Get the recipe
        recipe = get_recipe_by_name(self.recipe)

        # Get the neutral and oxidized charges
        neutral_charge = get_baseline_charge(record.identifier['inchi'])
        charged_charge = neutral_charge + (1 if oxidation_state == oxidation_state.REDUCED else -1)

        # Determine the starting point for relaxations, if required
        #  We'll use the neutral from the previous level for any computations
        if previous_level is None:
            neutral_start = charged_start = generate_inchi_and_xyz(record.identifier['inchi'])
        else:
            previous_step = get_recipe_by_name(previous_level.recipe)
            neutral_start = record.data[previous_step.geometry_level][OxidationState.NEUTRAL].xyz
            charged_start = record.data[previous_step.geometry_level].get(oxidation_state, OxidationState.NEUTRAL).xyz

        # Determine if any relaxations are needed
        geom_level = recipe.geometry_level
        if geom_level not in record.data or OxidationState.NEUTRAL not in record.data[geom_level]:
            required.append((geom_level, neutral_start, neutral_charge, None, True))
        if recipe.adiabatic and (geom_level not in record.data or oxidation_state not in record.data[geom_level]):
            required.append((geom_level, charged_start, charged_charge, None, True))

        # If any relaxations are triggered, then return the list now
        if len(required) > 0:
            return required

        # Determine if any single-point calculations are required
        neutral_geom_data = record.data[geom_level][OxidationState.NEUTRAL]
        if recipe.adiabatic:
            charged_geom_data = record.data[geom_level][oxidation_state]
        else:
            charged_geom_data = neutral_geom_data

        for state, data, chg in zip([OxidationState.NEUTRAL, oxidation_state],
                                    [neutral_geom_data, charged_geom_data],
                                    [neutral_charge, charged_charge]):
            if recipe.energy_level not in data.total_energy.get(state, {}):
                required.append((recipe.energy_level, data.xyz, chg, None, False))
            if recipe.solvent is not None and (
                    recipe.solvent not in data.total_energy_in_solvent.get(state, None) or
                    recipe.solvation_level not in data.total_energy_in_solvent[state][recipe.solvent]
            ):
                required.append((recipe.energy_level, data.xyz, chg, recipe.solvent, False))

        return required

    def get_eligible_molecules(self,
                               database: MoleculePropertyDB,
                               oxidation_state: OxidationState,
                               previous_level: Optional['FidelityLevel'] = None,
                               allowed_inchis: Optional[List[str]] = ()) -> Iterable[MoleculeData]:
        """Get a list of molecules that have not yet run at this level of fidelity

        If a `previous_level` is provided, this will produce a list of molecules that have already completed
        that level of fidelity.

        If a list of `allowed_inchis` is provided, only records that match these inchis will be passed

        Args:
            database: Connection to the Mongo database
            oxidation_state: Oxidation state used for the computation
            previous_level: Previous level of fidelity. Molecules that have not yet passed this level will be skipped
            allowed_inchis: List of allowed molecules
        Yields:
            Records for molecules that are ready to compute at this level
        """
        raise NotImplementedError()
