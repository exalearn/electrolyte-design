from moldesign.store.models import MoleculeData


def test_from_smiles():
    md = MoleculeData.from_identifier('C')
    assert md.identifiers['smiles'] == 'C'
    assert md.key == "VNWKTOKETHGBQD-UHFFFAOYSA-N"


def test_from_inchi():
    md = MoleculeData.from_identifier(inchi="InChI=1S/CH4/h1H4")
    assert md.identifiers['inchi'] == 'InChI=1S/CH4/h1H4'
    assert md.key == "VNWKTOKETHGBQD-UHFFFAOYSA-N"

