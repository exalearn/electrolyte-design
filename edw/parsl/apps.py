"""Workflow steps expressed as Parsl applications"""

import os
import logging
from gridfs import GridFS
from parsl import python_app
from parsl.dataflow.futures import AppFuture
from pymongo.collection import Collection
from edw.actions import geometry, nwchem, gaussian, mongo
from typing import List, Tuple, Any, Optional, Dict

__all__ = ['run_nwchem', 'run_gaussian', 'smiles_to_conformers',
           'match_future_with_inputs', 'store_and_validate_relaxation']

logger = logging.getLogger(__name__)


@python_app(executors=['local_threads'])
def match_future_with_inputs(inputs: Any, future) -> Tuple[Any, Any]:
    """Used as the last step of a workflow for easier tracking

    Args:
        inputs (Any): Some marker of the inputs to a workflow
        future (AppFuture): Output from the workflow
    Returns:
        inputs, result
    """
    logger.info('Running match_future_with_inpus')
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


@python_app()
def run_gaussian(gaussian_cmd: List[str], input_file: str,  run_dir: str) -> dict:
    """Run Gaussian

    Args:
        gaussian_cmd ([str]): Command used to invoke Gaussian
        input_file (str): Structure in XYZ format
        run_dir (str): Directory in which to run calculation
    Keyword Args:
        Passed to Gaussian input file creation
    Returns:
        (dict) Data from relaxation calculation including
            'input_file': Input file for the calculation
            'output_file': Complete output file for the calculation
            'successful': Whether the process completed successfully
    """

    # TODO (wardlt): Consider making this a Parsl Bash app

    # Run Gaussian
    #  TODO (wardlt): Can I replace this function with the QCArchive?
    result = gaussian.run_gaussian(input_file, 'gaussian', gaussian_cmd,
                                   run_dir=run_dir)

    # Read in the output file
    with open(result[1]) as fp:
        output_file = fp.read()

    # Record whether the calculation was successful
    successful = result[0].returncode == 0

    # Return the raw results
    return {
        'input_file': input_file,
        'output_file': output_file,
        'successful': successful
    }


@python_app
def run_nwchem(tag: str, input_file: str, nwchem_cmd: List[str]) -> dict:
    """Perform an NWChem calculation

    Writes the input file

    Args:
        tag (str): Name of the calculation, should be unique
        input_file (str): Input file to run
        nwchem_cmd ([str]): Command to issue NWChem
    Returns:
        (dict) Data from relaxation calculation including
            'input_file': Input file for the calculation
            'output_file': Complete output file for the calculation
            'successful': Whether the process completed successfully
    """

    # Create a run directory for this calculation
    run_dir = os.path.join('edw-run', 'nwchem', tag)
    os.makedirs(run_dir, exist_ok=True)

    # Invoke NWChem
    result = nwchem.run_nwchem(input_file, 'nw', nwchem_cmd, run_dir=run_dir)

    # Was the run successful
    successful = result[0].returncode == 0

    # Read the output file
    with open(result[1]) as fp:
        output_file = fp.read()
    return {
        'input_file': input_file,
        'output_file': output_file,
        'successful': successful
    }


@python_app(executors=['local_threads'])
def store_and_validate_relaxation(inchi_key: str,
                                  calc_name: str,
                                  geom_name: str,
                                  relax_result: dict,
                                  collection: Collection,
                                  gridfs: GridFS) \
        -> Optional[Tuple[Tuple[Any], Dict[str, Any]]]:
    """Process the outputs from a Gaussian relaxation:

    1. Check whether te calculation converged
    2. If so, store the result in MongoDB
    3. If not, make a new set of relaxation arguments

    Args:
        inchi_key (str): InChI key of molecule in question
        calc_name (str): Name to store calculation in the database
        geom_name (str): Name to store the geometry as
        relax_result (dict): Output of a Gaussian calculation
        collection (Collection): Connection to the MongoDB collection
        gridfs (GridFS): Connection the MongoDB GridFS store
    Returns:
        - ((Any)) New positional arguments to pass to pass to relaxation script
        - (dict): New keyword arguments to pass to relaxation script
    """
    logger.info(f'Storing results for {calc_name} calculation on {inchi_key}')

    # Store the calculation data in MongoDB
    mongo.add_calculation(collection, gridfs,
                          inchi_key, calc_name,
                          relax_result['input_file'],
                          relax_result['output_file'],
                          'gaussian')

    # Retrieve the output file
    output_file = relax_result['output_file']

    # Check if the relaxation completed successfully
    converged, new_structure = gaussian.validate_relaxation(output_file)

    # Store whether the calculation converged
    collection.update_one({'inchi_key': inchi_key},
                          {'$set':
                              {f'calculation.{calc_name}': {
                                  'validated': converged
                              }}})

    # If converged, store the result. We're done!
    if converged:
        logger.info(f'{calc_name} on {inchi_key} is converged. Storing geometry '
                    f'as {geom_name}')
        mongo.add_geometry(collection, inchi_key, geom_name, new_structure)
        return None
    else:
        # Use the new geometry as input to the function
        logger.info(f'{calc_name} on {inchi_key} did not converge')
        return (new_structure,), {}
