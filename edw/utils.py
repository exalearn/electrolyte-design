import contextlib
import os
from io import StringIO


@contextlib.contextmanager
def working_directory(path):
    """A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.

    Thanks to: http://code.activestate.com/recipes/576620-changedirectory-context-manager/

    Args:
        path (str): Desired working directory
    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(prev_cwd)


def mol_to_xyz(mol: str) -> str:
    """Convert a molecule block to XYZ format

    Args:
        mol (str): Molecule in mol format
    Returns:
        (str): Molecule rendered to XYZ
    """

    # Parse with RDKit
    lines = mol.split("\n")
    n_atoms = lines[3].split()[0]
    atoms = lines[3:4+int(n_atoms)]

    # Write out the XYZ file
    output = StringIO()
    print(n_atoms, file=output)
    for atom in atoms:
        values = atom.split()
        print(values[3], *values[:3], file=output)

    return output.getvalue()