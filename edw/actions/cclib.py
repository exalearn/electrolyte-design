"""Manipulating the output logs form CClib

Documentation of cclib record format: https://cclib.github.io/data.html"""

from pymatgen.core import Element
from io import StringIO
import numpy as np


def get_relaxed_structure(output: dict) -> str:
    """Get the relaxed structure in XYZ format

    Args:
        output (dict): CCLib output
    Returns:
        (str) Structure in XYZ format
    """

    # Get the element format
    Z = output['atoms']['elements']['numbers']
    symbols = [Element.from_Z(z).symbol for z in Z]

    # Get the coordinates
    coords = np.reshape(output['atoms']['coords']['3d'], (None, 3))

    # Print in XYZ format
    output = StringIO()
    print(len(symbols), file=output)
    print(file=output)
    for s, c in zip(symbols, coords):
        print(f'{s} {" ".join(map(str, c))}', file=output)
    return output.getvalue()
