"""Workflow steps related to NWChem"""

from edw.utils import working_directory

from pymatgen.io.gaussian import GaussianInput, GaussianOutput
from pymatgen.core import Molecule
from subprocess import run
import cclib
import json
import os


def make_input_file(mol: str, **kwargs):
    """Make a simple input file for Gaussian

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


def make_robust_relaxation_input(mol: str, functional: str = 'b3lyp',
                                 basis_set: str = '6-31G(2df,p)',
                                 charge=0, **kwargs) -> str:
    """Relaxation strategy that should deal with metastable molecules.

    Follows the procedure:
        1. SCF calculation where stability of wavefunctions is checked
        2. Frequency calculation, loading result from a checkpoint
        3. Optimization, where we read the force constants in Cartesian coordinates
        4. Frequency calculation, using the fully-relaxed geometry

    These are chained together in a multistep input file

    Args:
        mol (str): Unrelaxed molecule in XYZ format
        functional (str): QC method to use for calculation
        basis_set (str): Name of desired basis set
        charge (int): Charge on the molecule
    Returns:
        (str): Multi-step input file
    """

    # Parse the molecule
    mol_obj = Molecule.from_str(mol, 'xyz')
    n_electrons = mol_obj.nelectrons
    spin_multiplicity = 1 if n_electrons % 2 == 0 else 2

    # Make the first step: an SCF calculation
    link0_params = {'%chk': 'checkpoint.chk'}
    input_file = GaussianInput(mol_obj, functional=functional,
                               basis_set=basis_set, charge=charge,
                               spin_multiplicity=spin_multiplicity,
                               route_parameters={'stable': 'opt',
                                                 'scf': {'direct': None,
                                                         'xqc': None}},
                               link0_parameters=link0_params,
                               **kwargs).to_string().strip()

    # Make the second step (frequency calculation)
    #  The geometry gets read from the checkpoint
    next_step = GaussianInput(None, functional=functional, basis_set=basis_set,
                              spin_multiplicity=spin_multiplicity, charge=charge,
                              route_parameters={'freq': None,
                                                'geom': 'allcheck',
                                                'guess': 'check',
                                                'scf': {
                                                    'direct': None,
                                                    'xqc': None,
                                                    'maxcyc': '500'
                                                }},
                              link0_parameters=link0_params,
                              **kwargs).to_string().strip()

    input_file += f'\n--link1--\n{next_step}\n'

    # Make the geometry optimization step
    next_step = GaussianInput(None, functional=functional, basis_set=basis_set,
                              spin_multiplicity=spin_multiplicity, charge=charge,
                              route_parameters={'geom': 'allcheck',
                                                'guess': 'check',
                                                'opt': {
                                                    'rcfc': None,
                                                    'maxcyc': 100
                                                },
                                                'scf': {
                                                    'direct': None,
                                                    'xqc': None,
                                                    'maxcyc': '500'
                                                }},
                              link0_parameters=link0_params,
                              **kwargs).to_string().strip()

    input_file += f'\n--link1--\n{next_step}\n'

    # Make the frequency optimization step
    next_step = GaussianInput(None, functional=functional, basis_set=basis_set,
                              spin_multiplicity=spin_multiplicity, charge=charge,
                              route_parameters={'geom': 'allcheck',
                                                'guess': 'check',
                                                'freq': None,
                                                'scf': {
                                                    'direct': None,
                                                    'xqc': None,
                                                    'maxcyc': '500'
                                                }},
                              link0_parameters=link0_params,
                              **kwargs).to_string().strip()

    input_file += f'\n--link1--\n{next_step}\n'

    return input_file


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
