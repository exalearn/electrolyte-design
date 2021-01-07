from moldesign.simulate.specs import lookup_reference_energies
from moldesign.store.models import MoleculeData


def test_from_smiles():
    md = MoleculeData.from_identifier('C')
    assert md.identifiers['smiles'] == 'C'
    assert md.identifiers['inchi'] == 'InChI=1S/CH4/h1H4'
    assert md.key == "VNWKTOKETHGBQD-UHFFFAOYSA-N"


def test_from_inchi():
    md = MoleculeData.from_identifier(inchi="InChI=1S/CH4/h1H4")
    assert md.identifiers['inchi'] == 'InChI=1S/CH4/h1H4'
    assert md.key == "VNWKTOKETHGBQD-UHFFFAOYSA-N"


def test_atomization():
    md = MoleculeData.from_identifier("[C]")
    md.total_energies['neutral'] = {'small_basis': lookup_reference_energies('small_basis')['C']}
    md.vibrational_modes['neutral'] = {'small_basis': []}
    md.update_thermochem()
    assert md.atomization_energy['small_basis'] == 0
    assert md.atomization_energy['small_basis-no_zpe'] == 0
    assert md.zpes['neutral']['small_basis'] == 0


def test_redox():
    md = MoleculeData.from_identifier("[C]")
    md.total_energies['neutral'] = {'small_basis': lookup_reference_energies('small_basis')['C']}
    md.vibrational_modes['neutral'] = {'small_basis': []}
    md.total_energies['reduced'] = {'small_basis': lookup_reference_energies('small_basis')['C']}
    md.vibrational_modes['reduced'] = {'small_basis': []}
    md.update_thermochem()

    assert md.ea['vacuum']['small_basis-no_zpe'] == 0
