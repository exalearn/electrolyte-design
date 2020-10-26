"""General utilities"""
import logging
# TODO (wardlt): Consider breaking this into separate submodules
from typing import Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def generate_inchi_and_xyz(smiles: str) -> Tuple[str, str]:
    """Generate the XYZ coordinates and InChI string for a molecule using
    a standard procedure.

    We use the following deterministic procedure:

    1. Generates 3D coordinates with RDKit. Use a set random number seed
    2. Assign yet-undetermined stereochemistry based on the 3D geometry
    3. Generate an InCHi string for the molecules

    Args:
        smiles: SMILES string
    Returns:
        - InChI string for the molecule
        - XYZ coordinates for the molecule
    """

    # Generate 3D coordinates for the molecule
    mol = Chem.MolFromSmiles(smiles)
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

    return inchi, xyz
