from io import StringIO

from ase.io.xyz import simple_read_xyz
import numpy as np

from moldesign.simulate.functions import generate_inchi_and_xyz
from moldesign.simulate.init_geom import fix_cyclopropenyl


def test_cyclopropyl():
    smiles = 'Oc1c[cH+]1'
    inchi, xyz = generate_inchi_and_xyz(smiles, special_cases=True)

    # Make sure the initial ring is indeed buckled
    atoms = next(simple_read_xyz(StringIO(xyz), slice(None)))

    def _is_coplanar(x: np.ndarray):
        y = x[:3, :] - x[-1, :]
        return np.linalg.det(y) < 1e-4
    assert not _is_coplanar(atoms.positions[:4, :])

    # Attempt to flatten it back out
    xyz = fix_cyclopropenyl(xyz, 'Oc1c[cH+]1')
    atoms = next(simple_read_xyz(StringIO(xyz), slice(None)))
    with open('test.xyz', 'w') as fp:
        fp.write(xyz)
    assert _is_coplanar(atoms.positions[:4, :])
