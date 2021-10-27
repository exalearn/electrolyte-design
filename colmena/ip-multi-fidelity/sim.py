"""All of the simulation assays, simplified to have the same interface"""
from typing import Tuple, List, Optional
from qcelemental.models import OptimizationResult, AtomicResult, DriverEnum

from moldesign.simulate.functions import generate_inchi_and_xyz, relax_structure, run_single_point
from moldesign.simulate.specs import get_qcinput_specification
from moldesign.utils.chemistry import get_baseline_charge


def compute_vertical(smiles: str, spec_name: str = 'small_basis', n_nodes: int = 2) \
        -> Tuple[List[OptimizationResult], List[AtomicResult]]:
    """Perform the initial ionization potential computation of the vertical

    First relaxes the structure and then runs a single-point energy at the

    Args:
        smiles: SMILES string to evaluate
        spec_name: Quantum chemistry specification for the molecule
        n_nodes: Number of nodes per computation
    Returns:
        - Relax records for the neutral
        - Single point energy in oxidized state
    """

    # Make the initial geometry
    inchi, xyz = generate_inchi_and_xyz(smiles)
    init_charge = get_baseline_charge(smiles)

    # Make the compute spec
    compute_config = {'nnodes': n_nodes, 'cores_per_rank': 2}

    # Get the specification and make it more resilient
    spec, code = get_qcinput_specification(spec_name)
    if code == "nwchem":
        spec.keywords["dft__iterations"] = 150
        spec.keywords["geometry__noautoz"] = True

    # Compute the geometries
    neutral_xyz, _, neutral_relax = relax_structure(xyz, spec, charge=init_charge, code=code,
                                                    compute_config=compute_config)

    # Perform the single-point energy for the ionized geometry
    oxid_spe = run_single_point(neutral_xyz, DriverEnum.energy, spec, charge=init_charge + 1, code=code,
                                compute_config=compute_config)
    return [neutral_relax], [oxid_spe]


def compute_adiabatic(xyz: str, init_charge: int, spec_name: str = 'small_basis', n_nodes: int = 2) \
        -> Tuple[List[OptimizationResult], List[AtomicResult]]:
    """Compute the adiabatic ionization potential starting from a neutral geometry

    Just relaxes the structure in the oxidized state

    Args:
        xyz: XYZ of the neutral geometry
        init_charge: Charge of the nuetral molecule
        spec_name: Name of the computation spec to perform
        n_nodes: Number of nodes for the computation

    Returns:
        - Relaxation record
        - Not used, but to make the same interface for compute_vertical and
    """

    # Get the specification and make it more resilient
    spec, code = get_qcinput_specification(spec_name)
    if code == "nwchem":
        spec.keywords["dft__iterations"] = 150
        spec.keywords["geometry__noautoz"] = True

    # Compute the geometries
    compute_config = {'nnodes': n_nodes, 'cores_per_rank': 2}
    _, _, oxidized_relaxed = relax_structure(xyz, spec, charge=init_charge + 1, code=code,
                                             compute_config=compute_config)

    # Perform the solvation energy computations, if desired
    return [oxidized_relaxed], []


def compute_single_point(xyz: str, charge: int, solvent: Optional[str] = None,
                         spec_name: str = 'normal_basis', n_nodes: int = 2) \
        -> Tuple[List[OptimizationResult], List[AtomicResult]]:
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
    spec, code = get_qcinput_specification(spec_name, solvent)
    if code == "nwchem":
        spec.keywords["dft__iterations"] = 150
        spec.keywords["geometry__noautoz"] = True

    #
    spe_record = run_single_point(xyz, DriverEnum.energy, spec, charge=charge, code=code,
                                  compute_config=compute_config)

    return [], [spe_record]


if __name__ == "__main__":
    [neu_relax], [oxi_vert] = compute_vertical('C', n_nodes=1)
    print(neu_relax.energies[-1] - oxi_vert.return_result)
    [oxi_relax], _ = compute_adiabatic(neu_relax.final_molecule.to_string('xyz'), 0, n_nodes=1)
    print(neu_relax.energies[-1] - oxi_relax.energies[-1])

    # Run the high-fidelity model
    _, [neu_spe] = compute_single_point(neu_relax.final_molecule.to_string('xyz'),
                                        neu_relax.final_molecule.molecular_charge, n_nodes=1)
    _, [oxi_spe] = compute_single_point(oxi_relax.final_molecule.to_string('xyz'),
                                        oxi_relax.final_molecule.molecular_charge, n_nodes=1)
    print(neu_spe.return_result - oxi_spe.return_result)
