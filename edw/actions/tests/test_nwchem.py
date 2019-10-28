"""Simple test for NWChem"""

from edw.actions.geometry import smiles_to_conformers, mol_to_xyz
from edw.actions import nwchem, cclib
from tempfile import TemporaryDirectory
import shutil
import os

nwchem_cmd = ['mpirun', '-n', '1', 'nwchem']


def test_methane(tmpdir):
    methane = smiles_to_conformers('C', 1)[0]
    methane = mol_to_xyz(methane)
    input_file = nwchem.make_input_file(methane, task_kwargs=dict(theory='dft'))
    result = nwchem.run_nwchem(input_file, 'methane', nwchem_cmd, run_dir=tmpdir)
    assert result[0].returncode == 0
    assert os.path.isfile(result[1])
    assert os.path.isfile(result[2])

    output = nwchem.parse_output(result[1])
    assert isinstance(output[0], dict)
    assert isinstance(output[1], list)


def test_g4mp2_components():
    methane = mol_to_xyz(smiles_to_conformers('C', 1)[0])

    for charge in [0, 1]:
        task_configs, input_configs = nwchem.generate_g4mp2_configs(charge)
        for run_name in task_configs.keys():
            input_file = nwchem.make_input_file(methane, task_configs[run_name],
                                                input_configs[run_name])
            with TemporaryDirectory() as td:
                result = nwchem.run_nwchem(input_file, run_name, nwchem_cmd,
                                           run_dir=td)
                if result[0].returncode != 0:
                    shutil.copyfile(result[1], 'error.out')
                    shutil.copyfile(result[2], 'error.err')
                    with open('error.in', 'w') as fp:
                        fp.write(input_file)
                assert result[0].returncode == 0

