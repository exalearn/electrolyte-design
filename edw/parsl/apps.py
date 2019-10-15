"""Workflow steps expressed as Parsl applications"""

from parsl import python_app
from tempfile import TemporaryDirectory
from edw.actions import initial_geometry, nwchem, cclib
from concurrent.futures import as_completed
from typing import List, Tuple

__all__ = ['relax_structure', 'relax_conformers', 'smiles_to_conformers', 'collect_conformers']


# Simple wrappers over actions defined elsewhere
smiles_to_conformers = python_app(initial_geometry.smiles_to_conformers)


@python_app
def relax_structure(tag: str, structure: str, nwchem_cmd: List[str]) -> Tuple[str, str]:
    """Relax a structure with NWChem and return the output structure

    Args:
        tag (str): Name of the calculation
        structure (str): Structure in Mol format
        nwchem_cmd ([str]): Command to issue NWChem
    """

    with TemporaryDirectory(prefix=tag) as td:
        input_file = nwchem.make_input_file(structure, theory='dft')
        result = nwchem.run_nwchem(input_file, 'nw', nwchem_cmd, run_dir=td)

        # Parse the output
        cclib_out, pmg_out = nwchem.parse_output(result[1])
        strc = cclib.get_relaxed_structure(cclib_out)
        return strc


@python_app(executors=['local_threads'])
def relax_conformers(confs, nwchem_cmd):
    """Submit tasks to relax each conformer for a molecule

    Args:
        input_tuple ((str, [str])): Molecule smiles string and conformers in MOL format
        nwchem_cmd ([str]): Command used to launch NWChem
    Returns:
        ([AppFuture]): List of app futures
    """

    # Submit new jobs
    jobs = []
    for i, conf in enumerate(confs):
        tag = f'c{i}'
        jobs.append(relax_structure(tag, conf, nwchem_cmd))

    return jobs


@python_app(executors=['local_threads'])
def collect_conformers(inchi_key, conf_jobs) -> Tuple[str, List[str]]:
    """Collect the conformers for a certain calculation

    Args:
        inchi_key (str): InChI key for a molecule
        conf_jobs (AppFuture): Futures for the jobs running
    """

    relaxed_confs = [s.result() for s in as_completed(conf_jobs)]
    return inchi_key, relaxed_confs
