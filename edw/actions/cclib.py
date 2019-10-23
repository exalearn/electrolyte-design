"""Manipulating the output logs form CClib

Documentation of cclib record format: https://cclib.github.io/data.html"""

from cclib.io import ccread, CJSONWriter
from pymatgen.core import Element
from io import StringIO
import numpy as np
import json


def get_chemical_json(output_file: str) -> dict:
    """Parse a chemical json file from an output file

    Args:
        output_file (str): Content of an output file
    Returns:
        (dict) Content of the output file in Chemical JSON format
    """

    ccdata = ccread(StringIO(output_file))
    return json.loads(CJSONWriter(ccdata).generate_repr())


def get_relaxed_structure(output: dict) -> str:
    """Get the relaxed structure in XYZ format

    Args:
        output (dict): Output for a run in chemical JSON format
    Returns:
        (str) Structure in XYZ format
    """

    # Get the element format
    Z = output['atoms']['elements']['number']
    symbols = [Element.from_Z(z).symbol for z in Z]

    # Get the coordinates
    coords = np.reshape(output['atoms']['coords']['3d'], (-1, 3))

    # Print in XYZ format
    output = StringIO()
    print(len(symbols), file=output)
    print(file=output)
    for s, c in zip(symbols, coords):
        print(f'{s} {" ".join(map(str, c))}', file=output)
    return output.getvalue()
