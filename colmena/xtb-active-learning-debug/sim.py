from typing import Tuple, List, Optional
from qcelemental.models import OptimizationResult, AtomicResult


def _run_simulation(smiles: str, solvent: Optional[str], spec_name: str = 'xtb')\
        -> Tuple[List[OptimizationResult], List[AtomicResult]]:
    """Run the ionization potential computation

    Args:
        smiles: SMILES string to evaluate
        solvent: Name of the solvent
        spec: Quantum chemistry specification for the molecule
    Returns:
        Relax records for the neutral and ionized geometry
    """
    from moldesign.simulate.functions import generate_inchi_and_xyz, relax_structure, run_single_point
    from moldesign.simulate.specs import get_qcinput_specification
    from moldesign.utils.chemistry import get_baseline_charge
    from qcelemental.models import DriverEnum

    # Make the initial geometry
    inchi, xyz = generate_inchi_and_xyz(smiles)
    init_charge = get_baseline_charge(smiles)

    # Get the specification and make it more resilient
    spec, code = get_qcinput_specification(spec_name)

    # Compute the geometries
    neutral_xyz, _, neutral_relax = relax_structure(xyz, spec, charge=init_charge, code=code)
    oxidized_xyz, _, oxidized_relax = relax_structure(neutral_xyz, spec, charge=init_charge + 1, code=code)

    # Perform the solvation energy computations, if desired
    if solvent is None:
        return [neutral_relax, oxidized_relax], []

    solv_spec, code = get_qcinput_specification(spec_name, solvent=solvent)
    neutral_solv = run_single_point(neutral_xyz, DriverEnum.energy, solv_spec, charge=init_charge, code=code)
    oxidized_solv = run_single_point(oxidized_xyz, DriverEnum.energy, solv_spec, charge=init_charge + 1, code=code)
    return [neutral_relax, oxidized_relax], [neutral_solv, oxidized_solv]


def run_simulation(smiles: str, solvent: Optional[str] = None, dilation_factor: float = 1) \
        -> Tuple[List[OptimizationResult], List[AtomicResult]]:
    """Hack to make each execution run in a separate process. XTB or geoMETRIC is bad with file handles

    Args:
        smiles: SMILES string of molecule to evaluate
        solvent: Name of solvent to use when computing IP
        dilation_factor: A factor by which to expand the runtime of the simulation
            Used to enumlate longer simulations without spending CPU cycles
    """
    from concurrent.futures import ProcessPoolExecutor
    from time import perf_counter, sleep

    assert dilation_factor >= 1

    with ProcessPoolExecutor(max_workers=1) as exec:
        runtime = perf_counter()
        fut = exec.submit(_run_simulation, smiles, solvent)
        result = fut.result()
        runtime = perf_counter() - runtime

        # If the dilation factor is set,
        #  sleep until runtime * dilation_factor has elapsed
        if dilation_factor > 1:
            sleep(runtime * (dilation_factor - 1))
        return result
