"""Data models for storing molecular property data"""
from typing import Dict, List, Optional
from enum import Enum

from rdkit import Chem
from pydantic import BaseModel, Field
from qcelemental.models import Molecule

from moldesign.simulate.specs import levels


class OxidationState(str, Enum):
    """Names for different oxidation states"""
    NEUTRAL = "neutral"
    REDUCED = "reduced"
    OXIDIZED = "oxidized"


AccuracyLevel = Enum('AccuracyLevel', dict((x.upper(), x) for x in levels + ['g4mp2', 'experiment']))


_prop_desc = "for the molecule for different oxidation states at different levels of accuracy."


class MoleculeData(BaseModel):
    """Record for storing all data about a certain molecule"""

    # Describing the molecule
    key: str = Field(..., help="InChI key of the molecule. Used as a database key")
    identifiers: Dict[str, str] = Field(default_factory=dict, help="Different identifiers for the molecule,"
                                                                   " such as a SMILES string or CAS number")
    subsets: List[str] = Field(default_factory=list, help="Names of different subsets in which this molecule belongs")

    # Cross-references to other databases
    qcfractal: Dict[str, int] = Field(default_factory=dict, help='References to QCFractal records for this molecule')

    # Computed properties of the molecule
    geometries: Dict[OxidationState, Dict[AccuracyLevel, Molecule]] = Field(
        default_factory=dict, help="Relaxed geometries " + _prop_desc
    )
    total_energies: Dict[OxidationState, Dict[AccuracyLevel, float]] = Field(
        default_factory=dict, help="Total energy in Ha " + _prop_desc
    )
    vibrational_modes: Dict[OxidationState, Dict[AccuracyLevel, List[float]]] = Field(
        default_factory=dict, help="Vibrational temperatures in K " + _prop_desc
    )
    total_energies_in_solvents: Dict[OxidationState, Dict[str, Dict[AccuracyLevel, float]]] = Field(
        default_factory=dict, help="Total energy in Ha in different solvents " + _prop_desc
    )

    # Properties derived from the base computations
    zpes: Dict[OxidationState, Dict[AccuracyLevel, float]] = Field(
        default_factory=dict, help="Zero point energies in Ha " + _prop_desc
    )
    ip: Dict[OxidationState, Dict[str, Dict[AccuracyLevel, float]]] = Field(
        default_factory=dict, help="Ionization potential in V in different solvents " + _prop_desc
    )
    ea: Dict[OxidationState, Dict[str, Dict[AccuracyLevel, float]]] = Field(
        default_factory=dict, help="Electron affinity in V in different solvents " + _prop_desc
    )
    solvation_energy: Dict[OxidationState, Dict[str, Dict[AccuracyLevel, float]]] = Field(
        default_factory=dict, help="Solvation energy in kcal/mol for the molecule in different solvents " + _prop_desc
    )
    atomization_energy: Dict[AccuracyLevel, float] = Field(
        default_factory=dict, help="Ionization potential in Ha at different levels of accuracies"
    )

    @classmethod
    def from_identifier(cls, smiles: Optional[str] = None, inchi: Optional[str] = None) -> 'MoleculeData':
        """Initialize a data record given an identifier of the molecular structure

        You must supply exactly one of the arguments

        TODO (wardlt): Check for undefined stereochemistry?

        Args:
            smiles: SMILES string
            inchi: InChI string
        Returns:
            MoleculeData record with the key set properly
        """

        # Determine if the inputs are set properly
        is_set = [x is not None for x in [smiles, inchi]]
        if sum(is_set) != 1:
            raise ValueError('You must set exactly one of the input arguments')

        # Parse the input
        iden = mol = None
        if smiles is not None:
            mol = Chem.MolFromSmiles(smiles)
            iden = {'smiles': smiles}
        elif inchi is not None:
            mol = Chem.MolFromInchi(inchi)
            iden = {'inchi': inchi}

        # Compute the inchi key
        if mol is None:
            raise ValueError('Identifier did not parse correctly')
        key = Chem.MolToInchiKey(mol)
        return MoleculeData(key=key, identifiers=iden)
