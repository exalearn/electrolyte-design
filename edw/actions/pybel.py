"""Wrappers for Pybel utility functions"""
from pybel import readstring
import pybel


def get_rmsd(mol_a: str, mol_b: str) -> float:
    """Generate the RMSD between two molecules

    Args:
        mol_a (str): A molecule in XYZ format
        mol_b (str): A second molecule in XYZ format
    Return:
        (float) RMSD between the two molecules
    """

    # Parse the molecules
    obmol_a = readstring('xyz', mol_a).OBMol
    obmol_b = readstring('xyz', mol_b).OBMol

    # Compute the RMSD
    align = pybel.ob.OBAlign()
    align.SetRefMol(obmol_a)
    align.SetTargetMol(obmol_b)

    align.Align()
    return align.GetRMSD()
