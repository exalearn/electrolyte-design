"""Data models for storing molecular property data"""
from typing import Dict, List, Optional
from enum import Enum

from rdkit import Chem
from pydantic import BaseModel, Field
from qcelemental.physical_constants import constants

from moldesign.simulate.functions import subtract_reference_energies
from moldesign.simulate.specs import lookup_reference_energies
from moldesign.simulate.thermo import compute_zpe_from_freqs

f = constants.ureg.Quantity('96485.3329 A*s/mol')
e = constants.ureg.Quantity('1.602176634e-19 A*s')


class OxidationState(str, Enum):
    """Names for different oxidation states"""
    NEUTRAL = "neutral"
    REDUCED = "reduced"
    OXIDIZED = "oxidized"


_prop_desc = "for the molecule for different oxidation states at different levels of accuracy."


class MoleculeData(BaseModel):
    """Record for storing all data about a certain molecule"""

    # Describing the molecule
    key: str = Field(..., help="InChI key of the molecule. Used as a database key")
    identifier: Dict[str, str] = Field(default_factory=dict, help="Different identifiers for the molecule,"
                                                                  " such as a SMILES string or CAS number")
    subsets: List[str] = Field(default_factory=list, help="Names of different subsets in which this molecule belongs")

    # Cross-references to other databases
    qcfractal: Dict[str, int] = Field(default_factory=dict, help='References to QCFractal records for this molecule')

    # Computed properties of the molecule
    geometry: Dict[OxidationState, Dict[str, str]] = Field(
        default_factory=dict, help="Relaxed geometries " + _prop_desc
    )
    total_energy: Dict[OxidationState, Dict[str, float]] = Field(
        default_factory=dict, help="Electronic energy in Ha " + _prop_desc
    )
    vibrational_modes: Dict[OxidationState, Dict[str, List[float]]] = Field(
        default_factory=dict, help="Vibrational frequencies in Hz " + _prop_desc
    )
    total_energy_in_solvent: Dict[str, Dict[OxidationState, Dict[str, float]]] = Field(
        default_factory=dict, help="Electronic energy in Ha in different solvents " + _prop_desc
    )

    # Properties derived from the base computations
    zpe: Dict[OxidationState, Dict[str, float]] = Field(
        default_factory=dict, help="Zero point energies in Ha " + _prop_desc
    )
    ip: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        help="Ionization potential in V in different solvents for the molecule at different levels of accuracy."
    )
    ea: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        help="Electron affinity in V in different solvents for the molecule at different levels of accuracy."
    )
    solvation_energy: Dict[OxidationState, Dict[str, Dict[str, float]]] = Field(
        default_factory=dict, help="Solvation energy in Ha for the molecule in different solvents " + _prop_desc
    )
    atomization_energy: Dict[str, float] = Field(
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

    def update_thermochem(self, verbose: bool = False):
        """Compute the thermochemical properties, if possible

        Used to make any derived thermochemical properties, such as IP or atomization energy,
        are up-to-date in the database

        Args:
            verbose: Whether to print out log messages
        """

        self.update_zpes(verbose=verbose)
        self.update_atomization_energies()
        self.update_redox_properties()

    def update_zpes(self, verbose: bool = False):
        """Compute the zero-point energies from vibrational temperatures, if not already computed"""

        for state, freq_dict in self.vibrational_modes.items():
            # Add the dictionary, if needed
            if state not in self.zpe:
                self.zpe[state] = {}

            # Compute the ZPE, if needed
            for acc, freqs in freq_dict.items():
                if acc not in self.zpe[state]:
                    self.zpe[state][acc] = compute_zpe_from_freqs(freqs, verbose=verbose, name=self.key)

    def update_atomization_energies(self):
        """Compute atomization energy given total energies, if not already completed"""

        # Check that we have some ZPEs or total energies for neutral molecules
        if OxidationState.NEUTRAL not in self.total_energy:
            return

        # Get my molecule with hydrogens!
        mol = Chem.AddHs(self.mol)

        # Compute both with and without vibrational contributions
        for acc, elect_eng in self.total_energy[OxidationState.NEUTRAL].items():
            if acc in self.zpe.get(OxidationState.NEUTRAL, {}) and acc not in self.atomization_energy:
                zpe = self.zpe[OxidationState.NEUTRAL][acc]
                atom = subtract_reference_energies(elect_eng + zpe, mol, lookup_reference_energies(acc))
                self.atomization_energy[acc] = atom
            if acc + '-no_zpe' not in self.atomization_energy:
                atom = subtract_reference_energies(elect_eng, mol, lookup_reference_energies(acc))
                self.atomization_energy[acc + "-no_zpe"] = atom

    def update_redox_properties(self):
        """Compute redox properties, if not already completed"""

        # Store the neutral energies and neutral ZPEs
        if OxidationState.NEUTRAL not in self.total_energy:
            return
        neutral_energies = self.total_energy[OxidationState.NEUTRAL]
        neutral_zpes = self.zpe.get(OxidationState.NEUTRAL, {})

        # Compute them in vacuum
        for state, redox_dict in zip([OxidationState.REDUCED, OxidationState.OXIDIZED], [self.ea, self.ip]):
            # Get the dictioanry in which to write outputs
            if 'vacuum' not in redox_dict:
                redox_dict['vacuum'] = {}
            output = redox_dict['vacuum']

            # Get the charged energy and ZPE
            if state not in self.total_energy:
                continue
            charged_energies = self.total_energy[state]
            charged_zpes = self.zpe.get(state, {})
            p = -1 if state == OxidationState.REDUCED else 1

            # Compute the redox without vibrational contributions
            for acc, charged_eng in charged_energies.items():
                if acc not in neutral_energies:
                    continue
                g_chg = charged_eng - neutral_energies[acc]
                g_chg_u = constants.ureg.Quantity(g_chg * constants.hartree2kcalmol, 'kcal/mol')
                output[str(acc) + "-no_zpe"] = (p * g_chg_u / f).to("V").magnitude

            # Compute the redox with vibrational contributions
            for acc, charged_zpe in charged_zpes.items():
                if any(acc not in d for d in [neutral_energies, charged_energies, neutral_zpes]):
                    continue
                charged_eng = charged_energies[acc]
                g_chg = (charged_eng + charged_zpe) - (neutral_energies[acc] + neutral_zpes[acc])
                g_chg_u = constants.ureg.Quantity(g_chg * constants.hartree2kcalmol, 'kcal/mol')
                output[acc] = (p * g_chg_u / f).to("V").magnitude
