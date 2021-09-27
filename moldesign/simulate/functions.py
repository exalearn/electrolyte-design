"""Workflow-friendly functions for performing quantum chemistry

Each function is designed to have inputs and outputs
that are serializable and, ideally, possible to use
in languages besides Python
"""

from typing import Dict, Optional, Union

from qcelemental.models import OptimizationInput, Molecule, AtomicInput, OptimizationResult, AtomicResult, DriverEnum
from qcelemental.models.procedures import QCInputSpecification
from qcengine import compute_procedure, compute

from qcengine.config import TaskConfig


import logging
# TODO (wardlt): Consider breaking this into separate submodules
from typing import Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from moldesign.simulate.init_geom import fix_cyclopropenyl
from moldesign.utils.chemistry import parse_from_molecule_string

logger = logging.getLogger(__name__)
_code = 'nwchem'  # Default software used for QC


def generate_inchi_and_xyz(mol_string: str, special_cases: bool = True) -> Tuple[str, str]:
    """Generate the XYZ coordinates and InChI string for a molecule using
    a standard procedure.

    We use the following deterministic procedure:

    1. Generates 3D coordinates with RDKit. Use a set random number seed
    2. Assign yet-undetermined stereochemistry based on the 3D geometry
    3. Generate an InCHi string for the molecules

    We then have post-processing steps for common mistakes in generating geometries:

    1. Ensuring cyclopropenyl groups are planar

    Args:
        mol_string: SMILES or InChI string
        special_cases: Whether to perform the post-processing
    Returns:
        - InChI string for the molecule
        - XYZ coordinates for the molecule
    """

    # Generate 3D coordinates for the molecule
    mol = parse_from_molecule_string(mol_string)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)

    # Generate an InChI string with stereochemistry information
    AllChem.AssignStereochemistryFrom3D(mol)
    inchi = Chem.MolToInchi(mol)

    # Save geometry as 3D coordinates
    xyz = f"{mol.GetNumAtoms()}\n"
    xyz += inchi + "\n"
    conf = mol.GetConformer()
    for i, a in enumerate(mol.GetAtoms()):
        s = a.GetSymbol()
        c = conf.GetAtomPosition(i)
        xyz += f"{s} {c[0]} {c[1]} {c[2]}\n"

    # Special cases for odd kinds of molecules
    if special_cases:
        fix_cyclopropenyl(xyz, mol_string)

    return inchi, xyz


def relax_structure(xyz: str,
                    qc_config: QCInputSpecification,
                    charge: int = 0,
                    compute_config: Optional[Union[TaskConfig, Dict]] = None,
                    code: str = _code) -> Tuple[str, float, OptimizationResult]:
    """Compute the atomization energy of a molecule given the SMILES string

    Args:
        xyz (str): Structure of a molecule in XYZ format
        qc_config (dict): Quantum Chemistry configuration used for evaluating the energy
        charge (int): Charge of the molecule
        compute_config (TaskConfig): Configuration for the quantum chemistry code, such as parallelization settings
        code (str): Which QC code to use for the evaluation
    Returns:
        (str): Structure of the molecule
        (float): Electronic energy of this molecule
        (OptimizationResult): Full output from the calculation
    """

    # Parse the molecule
    mol = Molecule.from_data(xyz, dtype='xyz', molecular_charge=charge)

    # Run the relaxation
    if code == "nwchem":
        keywords = {"driver__maxiter": 100, "set__driver:linopt": 0}
        relax_code = "nwchemdriver"
    else:
        keywords = {"program": code}
        relax_code = "geometric"
    opt_input = OptimizationInput(input_specification=qc_config,
                                  initial_molecule=mol,
                                  keywords=keywords)
    res = compute_procedure(opt_input, relax_code, local_options=compute_config, raise_error=True)

    return res.final_molecule.to_string('xyz'), res.energies[-1], res


def compute_reference_energy(element: str, qc_config: QCInputSpecification,
                             n_open: int, code: str = _code) -> float:
    """Compute the energy of an isolated atom in vacuum

    Args:
        element (str): Symbol of the element
        qc_config (QCInputSpecification): Quantum Chemistry configuration used for evaluating he energy
        n_open (int): Number of open atomic orbitals
        code (str): Which QC code to use for the evaluation
    Returns:
        (float): Energy of the isolated atom
    """

    # Make the molecule
    xyz = f'1\n{element}\n{element} 0 0 0'
    mol = Molecule.from_data(xyz, dtype='xyz', molecular_multiplicity=n_open, molecular_charge=0)

    # Run the atomization energy calculation
    input_spec = AtomicInput(molecule=mol, driver='energy', **qc_config.dict(exclude={'driver'}))
    result = compute(input_spec, code, raise_error=True)

    return result.return_result


def run_single_point(xyz: str, driver: DriverEnum,
                     qc_config: QCInputSpecification,
                     charge: int = 0,
                     compute_config: Optional[Union[TaskConfig, Dict]] = None,
                     code: str = _code) -> AtomicResult:
    """Run a single point calculation

    Args:
        xyz: Structure in XYZ format
        driver: What type of property to compute: energy, gradient, hessian
        qc_config (dict): Quantum Chemistry configuration used for evaluating the energy
        charge (int): Charge of the molecule
        compute_config (TaskConfig): Configuration for the quantum chemistry code, such as parallelization settings
        code (str): Which QC code to use for the evaluation
    Returns:
        QCElemental-format result of the output
    """

    # Parse the molecule
    mol = Molecule.from_data(xyz, dtype="xyz", molecular_charge=charge)

    # Run the computation
    input_spec = AtomicInput(molecule=mol, driver=driver, **qc_config.dict(exclude={'driver'}))
    return compute(input_spec, code, local_options=compute_config, raise_error=True)
