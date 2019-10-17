from edw.actions import cclib
from pymatgen.io.xyz import XYZ
from pytest import fixture
import json
import os


@fixture
def cc_data() -> dict:
    path = os.path.dirname(__file__)
    outfile_path = os.path.join(path, 'files', 'g16', 'methane.json')
    with open(outfile_path) as fp:
        return json.load(fp)


def test_relaxed_structure(cc_data):
    xyz_file = cclib.get_relaxed_structure(cc_data)
    assert len(XYZ.from_string(xyz_file).molecule) == 5
