"""Workflow steps expressed as Parsl workflow elements"""

from parsl import python_app
from functools import partial
from tempfile import TemporaryDirectory
from edw.actions import initial_geometry, nwchem
from typing import List, Iterator, Tuple


# Wrap functions to keep track of the original inputs to a workflow
def _remember_tag_wrapper(func, tagged_inputs, *args, **kwargs):
    """Execute a function and return the value with the inputs as well"""
    tag, x = tagged_inputs
    return tag, func(x, *args, **kwargs)


smiles_to_conformers = python_app(partial(_remember_tag_wrapper,
                                          initial_geometry.smiles_to_conformers))


def generate_conformers(smiles: List[str]) -> Iterator[Tuple[str, List[str]]]:
    """A simple workflow to generate a series of conformers for a compound

    Args:
        smiles ([str[): SMILES string of a molecule
    Returns:
        ([str]) A list of ParslFutures, which are tuples of: SMILES, [conformers]
    """

    yield from map(lambda s: smiles_to_conformers((s, s), 16, relax=True), smiles)


@python_app
def relax_structure(tag: str, structure: str) -> Tuple[str, str]:
    """Relax a structure with NWChem and """

    with TemporaryDirectory(prefix=tag) as td:
        input_file = nwchem.make_input_file(structure, theory='dft')
        result = nwchem.run_nwchem(input_file, 'methane', ['mpirun', '-n',
                                                           '1', 'nwchem'],
                                   run_dir=td)

        conv, strc, energy = nwchem.read_relaxed_structure(result[1])
        return tag, strc
