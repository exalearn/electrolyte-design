from typing import Tuple, List, Optional
from qcelemental.models import OptimizationResult, AtomicResult, DriverEnum

from moldesign.simulate.functions import generate_inchi_and_xyz, relax_structure, run_single_point
from moldesign.simulate.specs import get_qcinput_specification
from moldesign.utils.chemistry import get_baseline_charge

from concurrent.futures import ProcessPoolExecutor
from time import perf_counter, sleep


def _run_with_delay(func, args, dilation_factor: float = 1):
    """Hack to make each execution run in a separate process. XTB or geoMETRIC is bad with file handles

    Args:
        func: Function to evaluate
        args: Input arguments
        dilation_factor: A factor by which to expand the runtime of the simulation
            Used to emulate longer simulations without spending CPU cycles
    """
    assert dilation_factor >= 1

    with ProcessPoolExecutor(max_workers=1) as exec:
        runtime = perf_counter()
        fut = exec.submit(func, *args)
        result = fut.result()
        runtime = perf_counter() - runtime

        # If the dilation factor is set,
        #  sleep until runtime * dilation_factor has elapsed
        if dilation_factor > 1:
            sleep(runtime * (dilation_factor - 1))
        return result


def _compute_vertical(smiles: str, solvent: Optional[str], spec_name: str = 'xtb') \
        -> Tuple[OptimizationResult, List[AtomicResult]]:
    """Run the ionization potential computation

    Args:
        smiles: SMILES string to evaluate
        solvent: Name of the solvent
        spec_name: Quantum chemistry specification for the molecule
    Returns:
        Relax records for the neutral and ionized geometry
    """

    # Make the initial geometry
    inchi, xyz = generate_inchi_and_xyz(smiles)
    init_charge = get_baseline_charge(smiles)

    # Get the specification and make it more resilient
    spec, code = get_qcinput_specification(spec_name)

    # Compute the geometries
    neutral_xyz, _, neutral_relax = relax_structure(xyz, spec, charge=init_charge, code=code)

    # Perform the solvation energy computations, if desired
    if solvent is None:
        oxid_spe = run_single_point(neutral_xyz, DriverEnum.energy, spec, charge=init_charge + 1, code=code)
        return neutral_relax, [oxid_spe]

    solv_spec, code = get_qcinput_specification(spec_name, solvent=solvent)
    neutral_solv = run_single_point(neutral_xyz, DriverEnum.energy, solv_spec, charge=init_charge, code=code)
    oxidized_solv = run_single_point(neutral_xyz, DriverEnum.energy, solv_spec, charge=init_charge + 1, code=code)
    return neutral_relax, [neutral_solv, oxidized_solv]


def compute_vertical(smiles: str, solvent: Optional[str] = None, dilation_factor: float = 1) \
        -> Tuple[OptimizationResult, List[AtomicResult]]:
    """Compute the vertical ionization potential

    Args:
        smiles: SMILES string of molecule to evaluate
        solvent: Name of solvent to use when computing IP
        dilation_factor: A factor by which to expand the runtime of the simulation
            Used to emulate longer simulations without spending CPU cycles
    """
    return _run_with_delay(_compute_vertical, (smiles, solvent), dilation_factor)


def _compute_adiabatic(xyz: str, init_charge: int, solvent: Optional[str], spec_name: str = 'xtb') \
        -> Tuple[OptimizationResult, List[AtomicResult]]:
    # Get the specification and make it more resilient
    spec, code = get_qcinput_specification(spec_name)

    # Compute the geometries
    oxid_xyz, _, oxidized_relaxed = relax_structure(xyz, spec, charge=init_charge + 1, code=code)

    # Perform the solvation energy computations, if desired
    if solvent is None:
        return oxidized_relaxed, []

    solv_spec, code = get_qcinput_specification(spec_name, solvent=solvent)
    oxidized_solv = run_single_point(oxid_xyz, DriverEnum.energy, solv_spec, charge=init_charge + 1, code=code)
    return oxidized_relaxed, [oxidized_solv]


def compute_adiabatic(xyz: str, init_charge: int, solvent: Optional[str] = None, dilation_factor: float = 1) \
        -> Tuple[OptimizationResult, List[AtomicResult]]:
    """Compute the adiabatic ionization potential

    Args:
        xyz: Starting geometry for the computation
        init_charge: Charge of the neutral molecule
        solvent: Name of a solvent, if desired
        dilation_factor: A factor by which to expand the runtime of the simulation
            Used to emulate longer simulations without spending CPU cycles
    """
    return _run_with_delay(_compute_adiabatic, (xyz, init_charge, solvent), dilation_factor)


if __name__ == "__main__":
    neu_relax, _ = compute_vertical('C')
    oxi_relax, _ = compute_adiabatic(neu_relax.final_molecule.to_string('xyz'), 0)
    print(neu_relax.energies[-1] - oxi_relax.energies[-1])
