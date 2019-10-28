"""Workflow steps related to NWChem

Our primary use case for NWChem is to perform static calculations,
such as computing the energy of a molecule or solvation energy.
"""

from edw.utils import working_directory

from pymatgen.io.nwchem import NwTask, NwInput, NwOutput
from pymatgen.core import Molecule
from subprocess import run, CompletedProcess
from typing import Tuple
import cclib
import json
import os


def generate_g4mp2_configs(charge: int = 0) -> Tuple[dict, dict]:
    """Generate the input file settings for G4MP2

    Some special settings are needed when running CCSD(T) on non-closed-shell systems.

    Args:
        charge (int): Charge on the system
    Returns:
        - (dict) Dictionary of the task settings for each component
        - (dict) Dictionary of the input file options for each component
    """
    # Default settings
    task_config = {
        'hf_g3lxp': {'theory': 'scf', 'basis_set': 'g3mp2largexp',
                     'operation': 'energy', 'basis_set_option': 'spherical'},
        'hf_pvtz': {'theory': 'scf', 'basis_set': 'g4mp2-aug-cc-pvtz',
                     'operation': 'energy', 'basis_set_option': 'spherical'},
        'hf_pvqz': {'theory': 'scf', 'basis_set': 'g4mp2-aug-cc-pvqz',
                    'operation': 'energy', 'basis_set_option': 'spherical'},
        'mp2_g3lxp': {'theory': 'mp2', 'basis_set': 'g3mp2largexp',
                      'operation': 'energy', 'basis_set_option': 'spherical',
                      'theory_directives': {'freeze': 'atomic'}},
        'ccsd(t)_small-basis': {'theory': 'tce', 'basis_set': '6-31G*',
                                'operation': 'energy',
                                'theory_directives': {'ccsd(t)': '',
                                                      'freeze': 'atomic'}}
    }
    input_config = dict((k, {}) for k in task_config.keys())

    # For a charge of 0, we need not do anything
    if charge == 0:
        return task_config, input_config

    # For non-zero charges, the MP2 and CCSD(T) calculations require changes:
    #  0. Specify the charge in the last description
    for value in task_config.values():
        value['charge'] = 1

    #  1. Specify the "doublet" multiplicity for all calculations
    for value in task_config.values():
        if value['theory'] == 'scf':
            # The setting is a "theory directive" when using SCF
            td = value.get('theory_directives', {})
            td['doublet'] = ''
            value['theory_directives'] = td
        else:
            # The setting is an "alternative directive" when not
            ad = value.get('alternate_directives', {})
            scf = ad.get('scf', {})
            scf['doublet'] = ''
            if value['theory'].startswith('mp2'):
                # Only needed for MP2. We use the TCE version of MP2 for CCSD(T)
                scf['uhf'] = ''
            ad['scf'] = scf
            value['alternate_directives'] = ad

    # 2. Change storage for 2eorb
    task_config['ccsd(t)_small-basis']['theory_directives']['2eorb'] = ''
    task_config['ccsd(t)_small-basis']['theory_directives']['2emet'] = '11'

    # 3. Add more memory for ccsd(t) calculation
    input_config['ccsd(t)_small-basis'] = {
        'memory_options': 'stack 1200 mb heap 100 mb global 600 mb'
    }

    return task_config, input_config


def make_input_file(mol: str, task_kwargs: dict, input_kwargs: dict = None) -> str:
    """Make input files for NWChem calculation

    Tasks are currently hard-wired to only perform a single task,
    although this is certainly not required for NWCHem

    This class uses Pymatgen's NWChem io module to make input files

    Args:
        mol (str): Molecule to be evaluated in XYZ format
        task_kwargs (dict): Keyword arguments for task creation
        input_kwargs (dict): Keyword arguments for the input file
    Returns:
        (str): Input file for the NWChem run
    """

    # Get defaults
    if input_kwargs is None:
        input_kwargs = {}

    # Parse the molecule
    mol_obj = Molecule.from_str(mol, 'xyz')

    # Generate the list of tasks
    task = NwTask.from_molecule(mol_obj, **task_kwargs)  # Just one for now

    # Make the input file
    nw_input = NwInput(mol_obj, tasks=[task], **input_kwargs)

    return str(nw_input)


def run_nwchem(input_file, job_name, executable, run_dir='.') \
        -> Tuple[CompletedProcess, str, str]:
    """Perform an NWChem calculation return the output file

    Assumes the calculation is to be started in the current working directory,
    and the job_name is unique. That is, if a files from the current `job_name`
    are in the current working directory, they are assumed to be restart files
    from the desired calculation.

    Args:
        input_file (str): Input file as a string
        job_name (str): Human-friendly name for the job
        executable ([str]): Invocation for NWChem (e.g., ['mpirun', 'nwchem'])
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
        with open(output_file, 'w') as fp, open(error_file, 'w') as fe:
            result = run(executable + [input_path], stdout=fp, stderr=fe)

        # Return output
        return result, os.path.join(run_dir, output_file), os.path.join(run_dir, error_file)


def parse_output(output_file):
    """Parse the output file

    Args:
        output_file (str): Path to the output file
    Returns:
        - (dict) Output from CCLib. ``None`` if parsing fails.
        - ([(dict])) Task information and errors
    """

    # Parse the output file with cclib
    cclib_out = None
    try:
        ccobj = cclib.io.ccopen(output_file).parse()
        cclib_out = json.loads(cclib.io.ccwrite(ccobj, 'json'))
    except BaseException:
        pass

    # Parse the task information with pymatgen
    accepted_keys = ['job_type', 'errors', 'task_time']
    output = NwOutput(output_file)
    task_out = []
    for task in output.data:
        subset = [(k, v) for k, v in task.items() if k in accepted_keys]
        task_out.append(subset)
    return cclib_out, task_out
