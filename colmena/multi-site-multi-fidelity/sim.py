"""Simulation steps with some platform-specific adjustments (e.g., setting up the cores per node"""
import hashlib
from typing import Optional, Any, Tuple, Dict, List
from qcelemental.models import AtomicResult, DriverEnum

from moldesign.simulate.functions import relax_structure, run_single_point
from moldesign.simulate.specs import get_qcinput_specification


def get_relaxation_args(xyz: str, charge: int, spec_name: str = 'small_basis', n_nodes: int = 2) \
        -> Tuple[List[Any], Dict[str, Any]]:
    """Get the function inputs to a relaxation computation

    Args:
        xyz: XYZ of the starting geometry
        charge: Charge of the molecule
        spec_name: Name of the computation spec to perform
        n_nodes: Number of nodes for the computation

    Returns:
        - Relaxation record
        - Not used, but to make the same interface for compute_vertical and
    """

    # Get the specification and make it more resilient
    spec, code = get_qcinput_specification(spec_name)
    if code == "nwchem":
        spec.keywords['dft__convergence__energy'] = 1e-7
        spec.keywords['dft__convergence__fast'] = True
        spec.keywords["dft__iterations"] = 150
        spec.keywords["driver__maxiter"] = 150
        spec.keywords["geometry__noautoz"] = True
        
        # Make sure to allow restarting
        spec.extras["allow_restarts"] = True
        runhash = hashlib.sha256(f'{xyz}_{charge}_{spec_name}'.encode()).hexdigest()[:12]
        spec.extras["scratch_name"] = f'nwc_{runhash}'

    # Set up compute configuration
    compute_config = {'nnodes': n_nodes, 'cores_per_rank': 2}

    return[xyz, spec], dict(charge=charge, code=code, compute_config=compute_config)


def get_single_point_args(xyz: str, charge: int, solvent: Optional[str] = None,
                          spec_name: str = 'normal_basis', n_nodes: int = 2) \
        -> Tuple[List[Any], Dict[str, Any]]:
    """Perform a single point energy computation, return results in same format as other assays

    Args:
        xyz: Molecular geometry
        charge: Molecular charge
        solvent: Name of the solvent, if desired
        spec_name: Name of the QC specification
        n_nodes: Number of nodes for the computation
    Returns:
        - Not used
        - Single point energy computation
    """

    # Get the specification and make it more resilient
    compute_config = {'nnodes': n_nodes, 'cores_per_rank': 2}
    if spec_name == 'diffuse_basis':
        compute_config['cores_per_rank'] = 8
    spec, code = get_qcinput_specification(spec_name, solvent)
    if code == "nwchem":
        # Reduce the accuracy needed to 1e-7
        spec.keywords['dft__convergence__energy'] = 1e-7
        spec.keywords['dft__convergence__fast'] = True
        spec.keywords["dft__iterations"] = 150
        spec.keywords["geometry__noautoz"] = True
        
        # Make sure to allow restarting
        spec.extras["allow_restarts"] = True
        runhash = hashlib.sha256(f'{xyz}_{charge}_{spec_name}_{solvent}'.encode()).hexdigest()[:12]
        spec.extras["scratch_name"] = f'nwc_{runhash}'

    return [xyz, DriverEnum.energy, spec], dict(charge=charge, code=code, compute_config=compute_config)
