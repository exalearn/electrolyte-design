"""Workflow steps related to NWChem"""

from edw.utils import working_directory

from pymatgen.io.gaussian import GaussianInput, GaussianOutput
from pymatgen.core import Molecule
from subprocess import run
import cclib
import json
import os


def make_input_file(mol: str, **kwargs):
    """Make input file for Gaussian file

    Keyword arguments are passed to the NwChem task creation

    Args:
        mol (str): Molecule to be evaluated in Mol format
    Returns:
        (str): Input file for the NWChem run
    """

    # Parse the molecule
    mol_obj = Molecule.from_str(mol, 'xyz')

    # Generate the list of tasks
    return GaussianInput(mol_obj, **kwargs).to_string()


def run_gaussian(input_file, job_name, executable, run_dir='.'):
    """Perform an Gaussian calculation return the output file

    Assumes the calculation is to be started in the current working directory,
    and the job_name is unique. That is, if a files from the current `job_name`
    are in the current working directory, they are assumed to be restart files
    from the desired calculation.

    Args:
        input_file (str): Input file as a string
        job_name (str): Human-friendly name for the job
        executable ([str]): Invocation for Gaussian (e.g., ['mpirun', 'nwchem'])
        run_dir (str): Directory in which to run the job
    Return:
        - (int): Return code from NWChem
        - (str): Path to the output file (will be named `{job_name}.out`)
        - (str): Path to the error file (will be named `{job_name}.err`)
    """

    with working_directory(run_dir):
        # Write the input file to disk
        input_path = f'{job_name}.in'
        with open(input_path, 'w') as fp:
            print(input_file, file=fp)

        # Start up NWChem
        output_file = f'{job_name}.out'
        error_file = f'{job_name}.err'
        with open(output_file, 'w') as fp, open(error_file, 'w') as fe, open(input_path) as fi:
            result = run(executable, stdin=fi, stdout=fp, stderr=fe)

        # Return output
        return result, os.path.join(run_dir, output_file), os.path.join(run_dir, error_file)


def parse_output(output_file):
    """Parse the output file

    Args:
        output_file (str): Path to the output file
    Returns:
        - (dict) Output from CCLib. ``None`` if parsing fails.
        - ([str]) List of errors
    """

    # Parse the output file with cclib
    cclib_out = None
    try:
        ccobj = cclib.io.ccopen(output_file).parse()
        cclib_out = json.loads(cclib.io.ccwrite(ccobj, 'json'))
    except BaseException:
        pass

    # Parse the error information with
    output = GaussianOutput(output_file)
    return cclib_out, output.errors
