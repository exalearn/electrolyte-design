from edw.actions import cclib
from pymatgen.io.xyz import XYZ
from pytest import fixture
import json
import os


path = os.path.dirname(__file__)


@fixture
def gaussian_output() -> str:
    outfile_path = os.path.join(path, 'files', 'g16', 'methane.out')
    with open(outfile_path) as fp:
        return fp.read()


@fixture
def chemical_json() -> dict:
    outfile_path = os.path.join(path, 'files', 'g16', 'methane.json')
    with open(outfile_path) as fp:
        return json.load(fp)


def test_parse_chemical_json(gaussian_output):
    cjson = cclib.get_chemical_json(gaussian_output)
    assert isinstance(cjson, dict)

    # Make sure it is JSON-serializable
    assert json.dumps(cjson).startswith('{')


def test_relaxed_structure(chemical_json):
    xyz_file = cclib.get_relaxed_structure(chemical_json)
    assert len(XYZ.from_string(xyz_file).molecule) == 5
