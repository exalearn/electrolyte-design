"""Workflow steps related to NWChem"""

from edw.utils import working_directory

from pymatgen.io.nwchem import NwTask, NwInput, NwOutput
from pymatgen.io.xyz import XYZ
from pymatgen.core import Molecule
from subprocess import run
from io import StringIO
import os


def _mol_to_xyz(mol: str) -> str:
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


def make_input_file(mol: str, **kwargs):
    """Make input files for NWChem calculation, currently hard-wired to relaxation

    Keyword arguments are passed to the NwChem task creation

    Args:
        mol (str): Molecule to be evaluated in Mol format
    Returns:
        (str): Input file for the NWChem run
    """

    # Parse the molecule
    mol_obj = Molecule.from_str(_mol_to_xyz(mol), 'xyz')

    # Generate the list of tasks
    task = NwTask.from_molecule(mol_obj, **kwargs)  # Just one for now

    # Make the input file
    nw_input = NwInput(mol_obj, tasks=[task])

    return str(nw_input)


def run_nwchem(input_file, job_name, executable, run_dir='.'):
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


def read_relaxed_structure(output_file):
    """Read the relaxed structure and energy of a molecule from a relaxation

    Args:
        output_file (str): Path to the output file
    Returns:
        - (bool) Whether the structure is converged
        - (str) Relaxed structure in XYZ format
        - (float) Total energy was structure
    """

    # Parse the output file
    output = NwOutput(output_file)

    # Make sure it is a relaxation calculation
    if len(output.data) != 1 \
            or output.data[0]['job_type'] != 'NWChem Geometry Optimization':
        raise ValueError('This calculation was not a geometry relaxation')
    relax_task_data = output.data[0]

    # Check for errors
    converged = len(relax_task_data['errors']) == 0

    # Get the molecular structure from the last timestep
    last_strc = str(XYZ(relax_task_data['molecules'][-1]))

    # Get the energy
    energy = relax_task_data['energies'][-1]

    return converged, last_strc, energy
