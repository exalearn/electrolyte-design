from edw.actions import gaussian, geometry
import os

path = os.path.dirname(__file__)

outfile_path = os.path.join(path, 'files', 'g16', 'methane.out')

def test_parse():
    cc_data, errors = gaussian.parse_output(outfile_path)
    assert isinstance(cc_data, dict)
    assert isinstance(errors, list)


def test_multistep_relaxation():
    mol = geometry.smiles_to_conformers('C', 1)[0]
    mol = geometry.mol_to_xyz(mol)

    input_file = gaussian.make_robust_relaxation_input(mol)
    assert len([x for x in input_file.split('\n') if 'link1' in x]) == 3


def test_validate_relaxation():
    with open(outfile_path) as fp:
        output_file = fp.read()
    is_converged, strc = gaussian.validate_relaxation(output_file)
    assert is_converged
    assert strc.startswith('5')
