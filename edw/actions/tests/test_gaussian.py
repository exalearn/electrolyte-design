from edw.actions import gaussian, geometry
import os

path = os.path.dirname(__file__)


def test_parse():
    outfile_path = os.path.join(path, 'files', 'g16', 'methane.out')
    cc_data, errors = gaussian.parse_output(outfile_path)
    assert isinstance(cc_data, dict)
    assert isinstance(errors, list)


def test_multistep_relaxation():
    mol = geometry.smiles_to_conformers('C', 1)[0]
    mol = geometry.mol_to_xyz(mol)

    input_file = gaussian.make_robust_relaxation_input(mol)
    assert len([x for x in input_file.split('\n') if 'link1' in x]) == 3
