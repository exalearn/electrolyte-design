"""Functions related to generating initial geometries for quantum chemistry codes"""

from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_conformers(smiles: str, n_conformers: int) -> List[str]:
    """Generate a series of conformers for a molecule

    Args:
        smiles (str): SMILES string for molecule of interest
        n_conformers (int): Number of conformers to generate
    Returns:
        ([str]): List of conformers in Mol format
    """

    # Make an RDK model of the molecules
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)

    # Generate conformers, with a pruning if RMS is less than 0.1 Angstrom
    ids = AllChem.EmbedMultipleConfs(m, numConfs=n_conformers,
                                     pruneRmsThresh=1)

    # Print out the conformers in XYZ format
    return [Chem.MolToMolBlock(m, confId=i) for i in ids]


def optimize_structure(mol: str) -> str:
    """Optimize the coordinates of a molecule using MMF94 forcefield

    Args:
        mol (str): String of molecule structure in Mol format
    Returns:
        (str): String of the relaxed structure in mol format
    """

    m = Chem.MolFromMolBlock(mol)
    while AllChem.MMFFOptimizeMolecule(m) == 1:
        continue
    return Chem.MolToMolBlock(m)
