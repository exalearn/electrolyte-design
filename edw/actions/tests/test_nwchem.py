"""Simple test for NWChem"""

from edw.actions.initial_geometry import smiles_to_conformers
from edw.actions import nwchem
import os


def test_methane(tmpdir):
    methane = smiles_to_conformers('C', 1)[0]
    input_file = nwchem.make_input_file(methane, theory='dft')
    result = nwchem.run_nwchem(input_file, 'methane', ['mpirun', '-n',
                                                       '1', 'nwchem'],
                               run_dir=tmpdir)
    assert result[0].returncode == 0
    assert os.path.isfile(result[1])
    assert os.path.isfile(result[2])

    output = nwchem.read_relaxed_structure(result[1])
