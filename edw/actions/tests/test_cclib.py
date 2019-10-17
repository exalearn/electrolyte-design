from edw.actions import cclib, gaussian
from pymatgen.io.xyz import XYZ
from pytest import fixture
import os


@fixture
def cc_data():
    path = os.path.dirname(__file__)
    outfile_path = os.path.join(path, 'files', 'g16', 'methane.out')
    cc_data, _ = gaussian.parse_output(outfile_path)
    return cc_data


def test_relaxed_structure(cc_data):
    xyz_file = cclib.get_relaxed_structure(cc_data)
    assert len(XYZ.from_string(xyz_file).molecule) == 5
