"""Workflow steps expressed as Parsl applications"""

from parsl import python_app
from tempfile import TemporaryDirectory
from edw.actions import geometry, nwchem, cclib, gaussian
from concurrent.futures import as_completed
from typing import List, Tuple, Any

__all__ = ['relax_nwchem', 'relax_gaussian', 'relax_conformers',
           'smiles_to_conformers', 'collect_conformers', 'match_future_with_inputs']


@python_app(executors=['local_threads'])
def match_future_with_inputs(inputs: Any, future) -> Tuple[Any, Any]:
    """Used as the last step of a workflow for easier tracking

    Args:
        inputs (Any): Some marker of the inputs to a workflow
        future (AppFuture): Output from the workflow
    Returns:
        inputs, result
    """
    return inputs, future


@python_app
def smiles_to_conformers(smiles: str, n: int) -> List[str]:
    """Generate initial conformers for a molecule

    Args:
        smiles (str): SMILES string of a molecule
        n (int): Number of conformers to generate
    Returns:
        ([str]) Conformers in XYZ format
    """
    confs = geometry.smiles_to_conformers(smiles, n_conformers=n)
    return [geometry.mol_to_xyz(m) for m in confs]


@python_app
def relax_gaussian(tag: str, structure: str, gaussian_cmd: List[str],
                   **kwargs) -> dict:
    """Use Gaussian to relax a structure and compute frequencies while at it

    Args:
        tag (str): Name of the calculation
        structure (str): Structure in XYZ format
        gaussian_cmd ([str]): Command to start Gaussian
    Keyword Args:
        Passed to Gaussian input file creation
    Returns:
        (dict) Data from relaxation calculation including
            'input_file': Input file for the calculation
            'output_file': Complete output file for the calculation
            'successful': Whether the process completed successfully
    """

    with TemporaryDirectory(prefix=tag) as td:
        input_file = gaussian.make_robust_relaxation_input(structure, **kwargs)
        result = gaussian.run_gaussian(input_file, 'gaussian', gaussian_cmd,
                                       run_dir=td)

        # Read in the output file
        with open(result[1]) as fp:
            output_file = fp.read()

        # Record whether the calculation was successful
        successful = result[0] == 0

        # Return the raw results
        return {
            'input_file': input_file,
            'output_file': output_file,
            'successful': successful
        }


@python_app
def relax_nwchem(tag: str, structure: str, nwchem_cmd: List[str]) -> str:
    """Relax a structure with NWChem and return the output structure

    Args:
        tag (str): Name of the calculation
        structure (str): Structure in XYZ format
        nwchem_cmd ([str]): Command to issue NWChem
    Returns:
        (str) Relaxed molecule in XYZ format
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
        jobs.append(relax_nwchem(tag, conf, nwchem_cmd))

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
