"""Simple test for NWChem"""

from edw.actions.geometry import smiles_to_conformers, mol_to_xyz
from edw.actions import nwchem
import os


def test_methane(tmpdir):
    methane = smiles_to_conformers('C', 1)[0]
    methane = mol_to_xyz(methane)
    input_file = nwchem.make_input_file(methane, theory='dft')
    result = nwchem.run_nwchem(input_file, 'methane', ['mpirun', '-n',
                                                       '1', 'nwchem'],
                               run_dir=tmpdir)
    assert result[0].returncode == 0
    assert os.path.isfile(result[1])
    assert os.path.isfile(result[2])

    output = nwchem.parse_output(result[1])
    assert isinstance(output[0], dict)
    assert isinstance(output[1], list)


def test_g4mp2_components():
    methane = mol_to_xyz(smiles_to_conformers('C', 1)[0])
    for key, kwargs in nwchem.g4mp2_configs.items():
        nwchem.make_input_file(methane, **kwargs)
