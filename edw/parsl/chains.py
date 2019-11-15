"""Functions that chain together application calls with functions to 
evaluate outputs and, if needed, trigger new simulations"""

from typing import Optional, List, Callable, Any, Tuple, Dict
from parsl.dataflow.futures import AppFuture
from parsl import python_app
from edw.actions import gaussian
from edw.parsl import apps
from pymongo.collection import Collection
from gridfs import GridFS
from functools import partial
import logging
import shutil
import os

logger = logging.getLogger(__name__)


@python_app(executors=['local_threads'])
def submit_or_none(new_inputs: Optional[Tuple[Tuple[Any], Dict[str, Any]]],
                   simulate_func: Callable,
                   cleanup_func: Optional[Callable] = None) -> Optional[AppFuture]:
    """Depending on the outcome of a Parsl app,
    submit another Parsl application or return "None"

    Args:
        new_inputs ((Any), dict): New positional arguments for the simulation function
        simulate_func (Callable): Simulation function to be called
        cleanup_func (Callable): Cleanup function to run otherwise
    Returns:
        (AppFuture) Future for the simulation function
    """
    if new_inputs is not None:
        logger.info(f'Resubmitting {simulate_func.__name__}')
        args, kwargs = new_inputs
        return simulate_func(*args, **kwargs)
    else:
        if cleanup_func is not None:
            cleanup_func()
        return None


def robust_relaxation(inchi_key: str, calc_name: str, gaussian_cmd: List[str],
                      structure: str, geom_name: str, collection: Collection,
                      gridfs: GridFS, **kwargs) -> Optional[AppFuture]:
    """Robust relaxation workchain
    
    Args:
        calc_name (str): Name of the calculation
        inchi_key (str): InChI key of structure being evaluated
        gaussian_cmd (str): Command used to launch Gaussian
        structure (str): XYZ format of the structure to be relaxed
        geom_name (str): Name with which to store geometry in database
        collection (Collection): Connection to the MongoDB collection
        gridfs (GridFS); Connection to the MongoDB GridFS store
    Keyword arguments are passed to :meth:`gaussian.make_robust_relaxation_input`
    Returns:
        (AppFuture) Either a future for the currently-executing tasks
            or None if the task successfully computed
    """

    # Make the tag for this calculation
    tag = f'{inchi_key}-{calc_name}'
    
    # Make a run directory for the calculation
    #  TODO (wardlt): Make this configurable?
    run_dir = os.path.join('gaussian-runs', tag)
    os.makedirs(run_dir, exist_ok=True)

    # Make the input file
    input_file = gaussian.make_robust_relaxation_input(structure, **kwargs)

    # Launch the Gaussian relaxation script
    result = apps.run_gaussian(gaussian_cmd, input_file, run_dir)

    # Process the output and store it in the database,
    #  returns True if the calculation is complete and
    #  no further calculations were resubmitted
    new_structure = apps.store_and_validate_relaxation(
        inchi_key, calc_name, geom_name, result,
        collection, gridfs)

    # Prepare to launch this relaxation again, recursively, by creating
    #  a function pointer to itself that takes only a new starting point
    rerun_fun = partial(robust_relaxation, inchi_key=inchi_key,
                        calc_name=calc_name, gaussian_cmd=gaussian_cmd,
                        geom_name=geom_name, collection=collection,
                        gridfs=gridfs, **kwargs)

    # Resubmit if needed
    return submit_or_none(new_structure, rerun_fun)
