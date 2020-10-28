from moldesign.simulate.functions import generate_inchi_and_xyz
from rdkit import Chem


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
