"""Tools for specifying a molecular design run"""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from moldesign.store.models import MoleculeData, OxidationState
from moldesign.store.recipes import apply_recipes


class ModelType(str, Enum):
    MPNN = "mpnn"
    SCHNET = "schnet"


class MultiFidelitySpecification(BaseModel):
    """Specification of a set of computation levels for redox properties of increasing accuracy"""

    levels: List[str] = Field(..., description="List of fidelity levels as names of redox recipes")

    def get_next_step(self, record: MoleculeData, oxidation_state: OxidationState) -> Optional[str]:
        """Get the next fidelity level for a certain molecule given what we know about it

        Args:
            record: Molecule to be evaluated
            oxidation_state: Which oxidation state we are evaluating
        Returns:
            The name of the next level of fidelity needed for this computation. ``None`` if all have been completed
        """

        # Make sure all of our evaluations are up-to-date
        apply_recipes(record)

        # Get the appropriate property we are looking through
        data = record.reduction_potential if oxidation_state == OxidationState.REDUCED else record.oxidation_potential

        # If the highest level is found, return None as we're done
        if self.levels[-1] in data:
            return None

        # Get the first level to be found in the molecule
        current_level = None
        for level in self.levels:
            if level in data:
                current_level = level
                break

        # If no level has been completed, return the first level
        if current_level is None:
            return self.levels[0]

        # Otherwise, return the next level in the chain
        return self.levels[self.levels.index(current_level) + 1]


class ModelCollection(BaseModel):
    """Collection of models used to predict a property"""

    required_level: str = Field(..., help='Required level of accuracy for a model')

    model_type: Optional[ModelType] = Field(None, help='Type of the model.')
    model_path: Optional[str] = Field(None, help='Pattern that matches the location of model files. '
                                                 '(e.g., model_dir/**/best.hdf5)')
    max_models: Optional[int] = Field(None, help='Maximum number of models to use for training or inference')
