"""Data models for storing molecular property data"""
from hashlib import sha1
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum

import numpy as np
from qcelemental.models.molecule import float_prep
from qcfractal.interface import FractalClient
from qcfractal.interface.models import OptimizationRecord, ResultRecord
from rdkit import Chem
from pydantic import BaseModel, Field
from qcelemental.models import Molecule, OptimizationResult, AtomicResult
from qcelemental.physical_constants import constants

from moldesign.simulate.specs import lookup_reference_energies, infer_specification_from_result
from moldesign.simulate.thermo import compute_zpe_from_freqs, subtract_reference_energies, compute_frequencies
from moldesign.utils.chemistry import get_baseline_charge

f = constants.ureg.Quantity('96485.3329 A*s/mol')
e = constants.ureg.Quantity('1.602176634e-19 A*s')


# Exceptions used by this class
class UnmatchedGeometry(Exception):
    """Error when a geometry is not found"""

    def __init__(self):
        super().__init__('Did not match the geometry')


# Simple definitions for things that define molecular properties, used for type definition readability
class OxidationState(str, Enum):
    """Names for different oxidation states"""
    NEUTRAL = "neutral"
    REDUCED = "reduced"
    OXIDIZED = "oxidized"

    @classmethod
    def from_charge(cls, charge: int, smiles: str) -> 'OxidationState':
        """Get the oxidation from the charge

        Args:
            charge: Charge state
            smiles: SMILES string of the molecule. Oxidation states are relative to the formal charge of the molecule
        Returns:
            Name of the charge state
        """

        net_charge = charge - get_baseline_charge(smiles)
        if net_charge == 0:
            return OxidationState.NEUTRAL
        elif net_charge == 1:
            return OxidationState.OXIDIZED
        elif net_charge == -1:
            return OxidationState.REDUCED
        else:
            raise ValueError(f'Unrecognized charge state, {charge}, for {smiles}')


def get_charge(state: Union[OxidationState, str]) -> int:
    """Get the charge associated with an oxidation state

    Args:
        state: Oxidation state
    """

    if state == OxidationState.NEUTRAL:
        return 0
    elif state == OxidationState.REDUCED:
        return -1
    elif state == OxidationState.OXIDIZED:
        return 1


AccuracyLevel = str
"""Name of an accuracy level. Just a string with no other validation requirements"""

SolventName = str
"""Name of the solvent"""

# Collections that store the properties of molecules and their geometries
_prop_desc = "for the geometry in different oxidation states at different levels of accuracy."


def get_hash(mol: Molecule) -> str:
    """Get a rotation and charge-invariant hash of the molecule.

    This is less sensitive than the hashing used by

    Args:
        mol: Molecule to be hashed
    Returns:
        Full hash string of the molecule
    """

    mol = Molecule.from_data(mol.to_string('xyz'), dtype='xyz', molecular_charge=0, orient=True)
    data: np.ndarray = float_prep(mol.geometry, 3)

    # Hash only the geometry (assume the masses etc are the same)
    my_hash = sha1()
    my_hash.update(data.tobytes())
    return my_hash.hexdigest()


class GeometryData(BaseModel):
    """Record for storing data about a certain geometry for a molecule"""

    # Storing the geometry
    xyz: str = Field(..., description="3D coordinates of the molecule in XYZ format")
    xyz_hash: str = Field(..., description="Hash of the 3D geometry produced by QCElemental."
                                           " We hash the oriented version of the molecule with a neutral charge (0).")

    # Provenance of the geometry
    fidelity: AccuracyLevel = Field(..., description="Level of the fidelity used to compute this molecule")
    oxidation_state: OxidationState = Field(..., description="Oxidation state used during geometry relaxation")

    # Computed properties of the molecule
    total_energy: Dict[OxidationState, Dict[AccuracyLevel, float]] = Field(
        default_factory=dict, help="Electronic energy in Ha " + _prop_desc
    )
    vibrational_modes: Dict[OxidationState, Dict[AccuracyLevel, List[float]]] = Field(
        default_factory=dict, help="Vibrational frequencies in Hz " + _prop_desc
    )
    total_energy_in_solvent: Dict[OxidationState, Dict[SolventName, Dict[AccuracyLevel, float]]] = Field(
        default_factory=dict, help="Electronic energy in Ha in different solvents " + _prop_desc
    )

    # Properties derived from computed properties
    zpe: Dict[OxidationState, Dict[AccuracyLevel, float]] = Field(
        default_factory=dict, help="Zero point energies in Ha " + _prop_desc
    )
    solvation_energy: Dict[OxidationState, Dict[SolventName, Dict[AccuracyLevel, float]]] = Field(
        default_factory=dict, help="Solvation energy in Ha for the molecule in different solvents " + _prop_desc
    )
    atomization_energy: Dict[AccuracyLevel, float] = Field(
        default_factory=dict, help="Atomization energy in Ha at different levels of accuracies"
    )

    def update_derived_properties(self, verbose: bool = True):
        """Update all derived properties for a molecule

        Includes thermochemistry and lookup hashes

        Args:
            verbose: Whether to print out log messages
        """

        self.xyz_hash = get_hash(Molecule.from_data(self.xyz, 'xyz'))
        self.update_thermochem(verbose=verbose)

    def update_thermochem(self, verbose: bool = False):
        """Compute the thermochemical properties using the available data

        Args:
            verbose: Whether to print out log messages
        """

        self.update_zpes(verbose=verbose)
        self.update_atomization_energies()
        self.update_solvation_energies()

    def update_zpes(self, verbose: bool = False):
        """Compute the zero-point energies from vibrational temperatures, if not already computed"""

        for state, freq_dict in self.vibrational_modes.items():
            # Add the dictionary, if needed
            if state not in self.zpe:
                self.zpe[state] = {}

            # Compute the ZPE, if needed
            for acc, freqs in freq_dict.items():
                if acc not in self.zpe[state]:
                    self.zpe[state][acc] = compute_zpe_from_freqs(freqs, verbose=verbose)

    def update_atomization_energies(self):
        """Compute atomization energy given total energies, if not already completed"""

        # Check that we have some ZPEs or total energies for neutral molecules
        if OxidationState.NEUTRAL not in self.total_energy:
            return

        # Get my molecule with hydrogens!
        mol = Molecule.from_data(self.xyz, "xyz")

        # Compute both with and without vibrational contributions
        for acc, elect_eng in self.total_energy[OxidationState.NEUTRAL].items():
            if acc in self.zpe.get(OxidationState.NEUTRAL, {}) and acc not in self.atomization_energy:
                zpe = self.zpe[OxidationState.NEUTRAL][acc]
                atom = subtract_reference_energies(elect_eng + zpe, mol, lookup_reference_energies(acc))
                self.atomization_energy[acc] = atom
            if acc + '-no_zpe' not in self.atomization_energy:
                atom = subtract_reference_energies(elect_eng, mol, lookup_reference_energies(acc))
                self.atomization_energy[acc + "-no_zpe"] = atom

    def update_solvation_energies(self):
        """Compute solvation energies given the available data"""

        # Loop over all oxidation states
        for state, vac_data in self.total_energy.items():
            if state not in self.total_energy_in_solvent:
                continue

            # Loop over all solvents
            for solvent, solv_data in self.total_energy_in_solvent[state].items():

                # Loop over all levels of accuracy
                for level, vac_eng in vac_data.items():
                    if level not in solv_data:
                        continue

                    # Add it to the solvation energy dictionary
                    if state not in self.solvation_energy:
                        self.solvation_energy[state] = {}
                    if solvent not in self.solvation_energy[state]:
                        self.solvation_energy[state][solvent] = {}
                    self.solvation_energy[state][solvent][level] = vac_eng - solv_data[level]


class MoleculeData(BaseModel):
    """Record for storing all summarized data about a certain molecule"""

    # Describing the molecule
    key: str = Field(..., help="InChI key of the molecule. Used as a database key")
    identifier: Dict[str, str] = Field(default_factory=dict, help="Different identifiers for the molecule,"
                                                                  " such as a SMILES string or CAS number")
    subsets: List[str] = Field(default_factory=list, help="Names of different subsets in which this molecule belongs")

    # Computed properties of the molecule
    data: Dict[AccuracyLevel, Dict[OxidationState, GeometryData]] = Field(
        default_factory=dict, help="Summary of molecular property calculations. Data are associated with "
                                   "a certain geometry of the molecule. Geometries are indexed by the level of theory "
                                   "used when relaxing the geometry and the oxidation state of the molecule."
    )

    # Properties derived from the base computations
    oxidation_potential: Dict[str, float] = Field(
        default_factory=dict,
        help="Absolute oxidation potential in V for the molecule in different conditions "
             "at different levels of accuracy."
    )
    reduction_potential: Dict[str, float] = Field(
        default_factory=dict,
        help="Absolute reduction potential in V for the molecule in different conditions "
             "at different levels of accuracy."
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
        output = MoleculeData(key=key, identifier=iden)
        output.add_all_identifiers()
        return output

    @property
    def mol(self) -> Chem.Mol:
        """Access the molecule as an RDKit object"""
        if 'smiles' in self.identifier:
            return Chem.MolFromSmiles(self.identifier['smiles'])
        elif 'inchi' in self.identifier:
            return Chem.MolFromInchi(self.identifier['inchi'])
        else:
            raise ValueError('No identifiers are compatible with RDKit')

    def add_all_identifiers(self):
        """Set all possible identifiers for a molecule"""

        # Get the data as a molecule
        mol = self.mol

        # Set the fields
        for name, func in [('smiles', Chem.MolToSmiles), ('inchi', Chem.MolToInchi)]:
            if name not in self.identifier:
                self.identifier[name] = func(mol)

    def match_geometry(self, mol: Molecule) -> Tuple[AccuracyLevel, OxidationState]:
        """Match the geometry to one in this record

        Args:
            mol: Molecule structure in XYZ format
        Returns:
            - Accuracy level used to compute this structure
            - Oxidation state of this structure
        Raises:
            KeyError if structure not found
        """

        # Get the hash of my molecule
        mol_hash = get_hash(mol)

        # See if we can find a match
        for level, geoms in self.data.items():
            for state, geom in geoms.items():
                if geom.xyz_hash == mol_hash:
                    return level, state
        raise UnmatchedGeometry()

    def add_geometry(self, relax_record: Union[OptimizationResult, OptimizationRecord],
                     spec_name: Optional[AccuracyLevel] = None,
                     overwrite: bool = False,
                     client: Optional[FractalClient] = None):
        """Add geometry to this record given a QCFractal

        Args:
            relax_record: Output from a relaxation computation with QCEngine
            spec_name: Name of the accuracy level. If ``None``, we will infer it from teh result
            overwrite: Whether to overwrite an existing record
            client: Connection to Fractal server, only needed for ``OptimizationRecord`` inputs
        """

        # Get the specification
        if spec_name is None:
            spec_name, solvent = infer_specification_from_result(relax_record, client)
            assert solvent is None, "We do not yet support relaxation in solvents."

        # Get the geometry
        if isinstance(relax_record, OptimizationRecord):
            geom = relax_record.get_final_molecule()
        else:
            geom = relax_record.final_molecule
        xyz = geom.to_string("xyz")
        xyz_hash = get_hash(geom)

        # Get the charge state for the geometry
        oxidation_state = OxidationState.from_charge(round(geom.molecular_charge), self.identifier['smiles'])

        # Check if the record already exists
        if not overwrite and spec_name in self.data and oxidation_state in self.data[spec_name]:
            raise ValueError(f'Already have {oxidation_state} geometry for {spec_name}')

        # Get the total energy
        total_energy = relax_record.energies[-1]

        # Make the record and save it
        entry = GeometryData(xyz=xyz, xyz_hash=xyz_hash, fidelity=spec_name, oxidation_state=oxidation_state,
                             total_energy={oxidation_state: {spec_name: total_energy}})
        entry.update_thermochem()
        if spec_name not in self.data:
            self.data[spec_name] = {}
        self.data[spec_name][oxidation_state] = entry

        # Check if the beginning matches another geometry,
        #   which allows us to define the total energy for that structure
        if isinstance(relax_record, OptimizationRecord):
            geom = relax_record.get_initial_molecule()
        else:
            geom = relax_record.initial_molecule

        try:
            init_level, init_state = self.match_geometry(geom)
        except UnmatchedGeometry:
            return

        # Add the energy
        geo_data = self.data[init_level][init_state]
        if oxidation_state not in geo_data.total_energy:
            geo_data.total_energy[oxidation_state] = dict()
        geo_data.total_energy[oxidation_state][spec_name] = relax_record.energies[0]

    def add_single_point(self, record: Union[AtomicResult, ResultRecord],
                         spec_name: Optional[AccuracyLevel] = None,
                         solvent_name: Optional[str] = None,
                         client: Optional[FractalClient] = None):
        """Add a single-point computation to the record

        Args:
            record: Record containing the data to be added
            spec_name: Specification used to compute this structure. If none provided, it is inferred from the
                specification in the result
            solvent_name: Name of the solvent, if known
            client: Connection to QCFractal server. Needed for result records
        """

        # Get the geometry
        if isinstance(record, AtomicResult):
            geom = record.molecule
        else:
            geom = record.get_molecule()

        # Match the geometry
        geom_level, geom_state = self.match_geometry(geom)
        geom_record = self.data[geom_level][geom_state]

        # Infer the method, if needed
        if spec_name is None:
            spec_name, solvent_name = infer_specification_from_result(record, client)

        # Get the oxidation state of the molecule used in this computation
        my_state = OxidationState.from_charge(round(geom.molecular_charge), self.identifier['smiles'])

        # All calculations compute the energy. Store it as appropriate
        total_energy = record.properties.return_energy
        if solvent_name is None:
            if my_state not in geom_record.total_energy:
                geom_record.total_energy[my_state] = {}
            geom_record.total_energy[my_state][spec_name] = total_energy
        else:
            if my_state not in geom_record.total_energy_in_solvent:
                geom_record.total_energy_in_solvent[my_state] = {}
            if solvent_name not in geom_record.total_energy_in_solvent[my_state]:
                geom_record.total_energy_in_solvent[my_state][solvent_name] = {}
            geom_record.total_energy_in_solvent[my_state][solvent_name][spec_name] = total_energy

        # Get the frequencies, if this was a Hessian computation
        if record.driver == "hessian":
            if solvent_name is not None:
                raise ValueError("We do not yet support Hessians in solvent in our data model!")
            freqs = compute_frequencies(record.return_result, geom).tolist()
            if my_state not in geom_record.vibrational_modes:
                geom_record.vibrational_modes[my_state] = {}
            geom_record.vibrational_modes[my_state][spec_name] = freqs

        # Update any thermodynamics
        geom_record.update_thermochem()

    def update_thermochem(self, verbose: bool = False):
        """Make sure all thermodynamic properties are computed

        Args:
            verbose: Whether to print status messages
        """

        for levels in self.data.values():
            for record in levels.values():
                record.update_thermochem(verbose)


# Definitions for schematics for how to compute geometries
class IonizationEnergyRecipe(BaseModel):
    """Defines the inputs needed to compute the ionization energy of a molecule"""

    # General information about the redox computation
    name: str = Field(..., help="Name of this recipe")
    solvent: Optional[str] = Field(None, help="Name of the solvent, if we are computing the total ")

    # Defining the accuracy level at which to compute the geometries for ion and neutral
    geometry_level: AccuracyLevel = Field(..., help="Accuracy level at which to compute the geometries")
    adiabatic: bool = Field(True, help="Whether to compute the adiabatic or the ")

    # Defining the level used to get the total energy, ZPE and solvation energy
    energy_level: AccuracyLevel = Field(..., help="Accuracy level used to compute the total energies")
    zpe_level: Optional[AccuracyLevel] = Field(None, help="Accuracy level used to compute the zero-point energy")
    solvation_level: Optional[AccuracyLevel] = Field(None, help="Accuracy level used to compute the solvation energy")

    def get_required_data(self, oxidation_state: Union[str, OxidationState]) -> List[str]:
        """List of data fields required for this computation to complete

        Args:
            oxidation_state: Target oxidation state
        """

        # Mark the 3D geometries used for the neutral and ionic properties
        neutral_rec = f'data.{self.geometry_level}.neutral'
        if self.adiabatic:
            charged_rec = f'data.{self.geometry_level}.{oxidation_state}'
        else:
            charged_rec = neutral_rec

        # Add in the required geometries
        output = [f'{x}.xyz' for x in [neutral_rec, charged_rec]]

        # Add in the required total energies
        output.extend([
            f'{neutral_rec}.total_energy.neutral.{self.energy_level}',
            f'{charged_rec}.total_energy.{oxidation_state}.{self.energy_level}',
        ])

        # Add in ZPEs, if needed
        if self.zpe_level is not None:
            output.extend([
                f'{neutral_rec}.zpe.neutral.{self.energy_level}',
                f'{charged_rec}.zpe.{oxidation_state}.{self.energy_level}',
            ])

        # Add in the solvation energies, if needed
        if self.solvent is not None and self.solvation_level is not None:
            output.extend([
                f'{neutral_rec}.total_energy_in_solvent.neutral.{self.solvent}.{self.energy_level}',
                f'{charged_rec}.total_energy_in_solvent.{oxidation_state}.{self.solvent}.{self.energy_level}',
            ])
        return output

    def compute_ionization_potential(self, mol_data: MoleculeData,
                                     oxidation_state: Union[str, OxidationState]) -> float:
        """Compute and store the ionization energy for a certain molecule

        Args:
            mol_data: Data recording a certain molecule
            oxidation_state: Oxidation state to evaluate
        Returns:
            Absolute ionization potential in V
        """

        # Get the geometry used for the neutral molecule
        if self.geometry_level not in mol_data.data:
            raise ValueError(f'No geometries available at level: {self.geometry_level}')
        geometry_data = mol_data.data[self.geometry_level]
        neutral_geometry = geometry_data[OxidationState.NEUTRAL]

        # Get the geometry used for the ion
        ionized_geometry = neutral_geometry
        if self.adiabatic:
            if oxidation_state not in geometry_data:
                raise ValueError(f'Lack {oxidation_state} geometry at level: {self.geometry_level}')
            ionized_geometry = geometry_data[oxidation_state]

        # Get the difference in the total energies in vacuum
        if self.energy_level not in neutral_geometry.total_energy[OxidationState.NEUTRAL]:
            raise ValueError(f'Total energy not available at {self.energy_level} for neutral molecule')
        if self.energy_level not in ionized_geometry.total_energy[oxidation_state]:
            raise ValueError(f'Total energy not available at {self.energy_level} for {oxidation_state} molecule')
        delta_g = ionized_geometry.total_energy[oxidation_state][self.energy_level] - \
            neutral_geometry.total_energy[OxidationState.NEUTRAL][self.energy_level]

        # Adjust with the ZPEs, if required
        if self.zpe_level is not None:
            if self.zpe_level not in neutral_geometry.zpe[OxidationState.NEUTRAL]:
                raise ValueError(f'ZPE not available at {self.energy_level} for neutral molecule')
            if self.zpe_level not in ionized_geometry.zpe[oxidation_state]:
                raise ValueError(f'ZPE not available at {self.energy_level} for {oxidation_state} molecule')
            delta_g += ionized_geometry.zpe[oxidation_state][self.energy_level] - \
                neutral_geometry.zpe[OxidationState.NEUTRAL][self.energy_level]

        # Adjust with solvation energy, if required
        if self.solvation_level is not None and self.solvent is not None:
            if self.solvation_level not in \
                    neutral_geometry.total_energy_in_solvent[OxidationState.NEUTRAL][self.solvent]:
                raise ValueError(f'Energy not available in {self.solvent} at {self.energy_level} for neutral molecule')
            if self.solvation_level not in ionized_geometry.total_energy_in_solvent[oxidation_state][self.solvent]:
                raise ValueError(f'Energy not available in {self.solvent} at {self.energy_level}'
                                 f' for {oxidation_state} molecule')
            delta_g += \
                ionized_geometry.solvation_energy[oxidation_state][self.solvent][self.solvation_level] - \
                neutral_geometry.solvation_energy[OxidationState.NEUTRAL][self.solvent][self.solvation_level]

        # Compute the absolute potential using the Nerst equation
        n = get_charge(oxidation_state)
        delta_g = constants.ureg.Quantity(delta_g * constants.hartree2kcalmol, "kcal/mol")
        potential = (delta_g / f / n).to("V").magnitude

        # Add it to the molecule data
        if oxidation_state == OxidationState.REDUCED:
            mol_data.reduction_potential[self.name] = potential
        else:
            mol_data.oxidation_potential[self.name] = potential
        return potential
