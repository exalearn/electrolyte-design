import json
import os

from qcelemental.models import Molecule
from pytest import fixture
from rdkit import Chem
import numpy as np

from moldesign.simulate.functions import generate_inchi_and_xyz
from moldesign.simulate.thermo import compute_zpe, subtract_reference_energies
from moldesign.simulate.specs import lookup_reference_energies

_path = os.path.dirname(__file__)

_mol = """13
1 2 C1C2CC1C2_oxidized_g16
C                     1.857861000440    -0.579991997377     0.161639999514
C                    -0.863332002464     0.042333002080     0.554971001285
C                     0.000001000145     1.137358001800     0.000878000244
C                     0.863705998458     0.043894000767    -0.555092997799
C                    -1.858058002532    -0.579573000153    -0.162452000776
H                     2.066767002575    -0.321588002522     1.195692999933
H                     2.490256998939    -1.329512999875    -0.303199999344
H                    -0.645723998606    -0.289120001380     1.565589997483
H                    -0.471766002486     1.745112000926    -0.768098001317
H                     0.471432001706     1.744150999236     0.770821999218
H                     0.646742998118    -0.285166999990    -1.566638001199
H                    -2.490860001663    -1.329055002291     0.301908997454
H                    -2.067919000195    -0.318938999615    -1.195744002032"""


@fixture()
def mol() -> Molecule:
    return Molecule.from_data(_mol)


@fixture()
def hess() -> np.ndarray:
    with open(os.path.join(_path, 'hess.json')) as fp:
        return np.array(json.load(fp))


def test_zpe(mol, hess):
    zpe = compute_zpe(hess, mol)
    print(zpe)


def test_subtract_reference_energies():
    # Make a RDKit and QCEngine representation of a molecule
    _, xyz = generate_inchi_and_xyz('C')
    molr = Chem.MolFromSmiles('C')
    molr = Chem.AddHs(molr)
    molq = Molecule.from_data(xyz, 'xyz')

    # Get the desired answer
    my_ref = lookup_reference_energies('small_basis')
    actual = my_ref['C'] + 4 * my_ref['H']

    # Check it works with either RDKit or QCElemental inputs
    assert subtract_reference_energies(0, molr, my_ref) == -actual
    assert subtract_reference_energies(0, molq, my_ref) == -actual
