from rdkit import Chem
from qcelemental.models import Molecule

from moldesign.simulate.functions import generate_inchi_and_xyz, subtract_reference_energies
from moldesign.simulate.specs import lookup_reference_energies


def test_generate():
    # Test with a simple molecule
    inchi, xyz = generate_inchi_and_xyz('C')
    assert xyz.startswith('5')
    assert inchi == 'InChI=1S/CH4/h1H4'

    # Test with a molecule that has defined stereochemistry
    inchi, xyz = generate_inchi_and_xyz("C/C=C\\C")
    assert inchi == "InChI=1S/C4H8/c1-3-4-2/h3-4H,1-2H3/b4-3-"

    # Change the stereo chemistry
    inchi_man_iso = Chem.MolToInchi(Chem.MolFromSmiles("C/C=C/C"))
    inchi_iso, xyz_iso = generate_inchi_and_xyz("C/C=C/C")
    assert inchi_man_iso == inchi_iso
    assert inchi != inchi_iso

    # Test with a molecule w/o defined sterochemistry
    inchi_undef, xyz = generate_inchi_and_xyz("CC=CC")
    assert inchi_undef == inchi_man_iso  # Make sure it gets the lower-energy isomer


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
