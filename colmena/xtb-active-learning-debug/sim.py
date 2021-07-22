from typing import Tuple, List
from qcelemental.models import OptimizationResult, AtomicResult


def _run_simulation(smiles: str) -> Tuple[List[OptimizationResult], List[AtomicResult]]:
    """Run the ionization potential computation

    Args:
        smiles: SMILES string to evaluate
    Returns:
        Relax records for the neutral and ionized geometry
    """
    from moldesign.simulate.functions import generate_inchi_and_xyz, relax_structure, run_single_point
    from moldesign.simulate.specs import get_qcinput_specification

    # Make the initial geometry
    inchi, xyz = generate_inchi_and_xyz(smiles)

    # Get the specification and make it more resilient
    spec, code = get_qcinput_specification('xtb')

    # Compute the neutral geometry and hessian
    neutral_xyz, _, neutral_relax = relax_structure(xyz, spec, charge=0, code=code)
    # neutral_hessian = run_single_point(neutral_xyz, DriverEnum.hessian, spec, charge=0, code=code)

    # Compute the relaxed geometry
    oxidized_xyz, _, oxidized_relax = relax_structure(neutral_xyz, spec, charge=1, code=code)
    # oxidized_hessian = run_single_point(oxidized_xyz, DriverEnum.hessian, spec, charge=1, code=code)
    return [neutral_relax, oxidized_relax], []  # , [neutral_hessian, oxidized_hessian]


def run_simulation(smiles: str, dilation_factor: float = 1) -> Tuple[List[OptimizationResult], List[AtomicResult]]:
    """Hack to make each execution run in a separate process. XTB or geoMETRIC is bad with file handles

    Args:
        smiles: SMILES string of molecule to evaluate
        dilation_factor: A factor by which to expand the runtime of the simulation
            Used to enumlate longer simulations without spending CPU cycles
    """
    from concurrent.futures import ProcessPoolExecutor
    from time import perf_counter, sleep

    assert dilation_factor >= 1

    with ProcessPoolExecutor(max_workers=1) as exec:
        runtime = perf_counter()
        fut = exec.submit(_run_simulation, smiles)
        result = fut.result()
        runtime = perf_counter() - runtime

        # If the dilation factor is set,
        #  sleep until runtime * dilation_factor has elapsed
        if dilation_factor > 1:
            sleep(runtime * (dilation_factor - 1))
        return result
