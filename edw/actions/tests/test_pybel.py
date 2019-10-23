from edw.actions import geometry, pybel
from math import isclose


def test_rmsd():
    mol = geometry.mol_to_xyz(geometry.smiles_to_conformers('C', 1)[0])
    assert isclose(pybel.get_rmsd(mol, mol), 0)
