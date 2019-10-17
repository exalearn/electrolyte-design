from edw.actions import gaussian
import os

path = os.path.dirname(__file__)


def test_parse():
    outfile_path = os.path.join(path, 'files', 'g16', 'methane.out')
    cc_data, errors = gaussian.parse_output(outfile_path)
    assert isinstance(cc_data, dict)
    assert isinstance(errors, list)
