"""Data models for storing molecular property data"""
from typing import Dict, List, Optional
from collections import defaultdict
from enum import Enum

from rdkit import Chem
from pydantic import BaseModel, Field
from qcelemental.models import Molecule

from moldesign.simulate.functions import subtract_reference_energies
from moldesign.simulate.specs import levels, lookup_reference_energies
from moldesign.simulate.thermo import compute_zpe_from_freqs


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
        default_factory=lambda: defaultdict(dict), help="Relaxed geometries " + _prop_desc
    )
    total_energies: Dict[OxidationState, Dict[AccuracyLevel, float]] = Field(
        default_factory=lambda: defaultdict(dict), help="Total energy in Ha " + _prop_desc
    )
    vibrational_modes: Dict[OxidationState, Dict[AccuracyLevel, List[float]]] = Field(
        default_factory=lambda: defaultdict(dict), help="Vibrational frequencies in Hz " + _prop_desc
    )
    total_energies_in_solvents: Dict[OxidationState, Dict[str, Dict[AccuracyLevel, float]]] = Field(
        default_factory=lambda: defaultdict(dict), help="Total energy in Ha in different solvents " + _prop_desc
    )

    # Properties derived from the base computations
    zpes: Dict[OxidationState, Dict[AccuracyLevel, float]] = Field(
        default_factory=lambda: defaultdict(dict), help="Zero point energies in Ha " + _prop_desc
    )
    ip: Dict[OxidationState, Dict[str, Dict[AccuracyLevel, float]]] = Field(
        default_factory=lambda: defaultdict(dict),
        help="Ionization potential in V in different solvents " + _prop_desc
    )
    ea: Dict[OxidationState, Dict[str, Dict[AccuracyLevel, float]]] = Field(
        default_factory=lambda: defaultdict(dict), help="Electron affinity in V in different solvents " + _prop_desc
    )
    solvation_energy: Dict[OxidationState, Dict[str, Dict[AccuracyLevel, float]]] = Field(
        default_factory=lambda: defaultdict(dict),
        help="Solvation energy in kcal/mol for the molecule in different solvents " + _prop_desc
    )
    atomization_energy: Dict[AccuracyLevel, float] = Field(
        default_factory=lambda: defaultdict(dict), help="Ionization potential in Ha at different levels of accuracies"
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

        # Output the molecule with all identifiers set
        output = MoleculeData(key=key, identifiers=iden)
        output.add_all_identifiers()
        return output

    @property
    def mol(self) -> Chem.Mol:
        """Access the molecule as an RDKit object"""
        if 'smiles' in self.identifiers:
            return Chem.MolFromSmiles(self.identifiers['smiles'])
        elif 'inchi' in self.identifiers:
            return Chem.MolFromInchi(self.identifiers['inchi'])
        else:
            raise ValueError('No identifiers are compatible with RDKit')

    def add_all_identifiers(self):
        """Set all possible identifiers for a molecule"""

        # Get the data as a molecule
        mol = self.mol

        # Set the fields
        for name, func in [('smiles', Chem.MolToSmiles), ('inchi', Chem.MolToInchi)]:
            if name not in self.identifiers:
                self.identifiers[name] = func(mol)

    def update_thermochem(self):
        """Compute the thermochemical properties, if possible

        Used to make any derived thermochemical properties, such as IP or atomization energy,
        are up-to-date in the database"""

        self.update_zpes()
        self.update_atomization_energies()

    def update_zpes(self):
        """Compute the zero-point energies from vibrational temperatures, if not already computed"""

        for state, freq_dict in self.vibrational_modes.items():
            # Add the dictionary, if needed
            if state not in self.zpes:
                self.zpes[state] = {}

            # Compute the ZPE, if needed
            for acc, freqs in freq_dict.items():
                if acc not in self.zpes[state]:
                    self.zpes[state][acc] = compute_zpe_from_freqs(freqs, verbose=True, name=self.key)

    def update_atomization_energies(self):
        """Compute atomization energy given total energies, if not already completed"""

        # Check that we have some ZPEs or total energies for neutral molecules
        if OxidationState.NEUTRAL not in self.total_energies or \
           OxidationState.NEUTRAL not in self.zpes:
            return

        # Get my molecule with hydrogens!
        mol = Chem.AddHs(self.mol)

        for acc, elect_eng in self.total_energies[OxidationState.NEUTRAL].items():
            if acc in self.zpes[OxidationState.NEUTRAL] and acc not in self.atomization_energy:
                zpe = self.zpes[OxidationState.NEUTRAL][acc]
                atom = subtract_reference_energies(elect_eng + zpe, mol, lookup_reference_energies(acc))
                self.atomization_energy[acc] = atom
