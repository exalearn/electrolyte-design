"""Specifications for different levels of computational chemistry"""
from typing import Optional, Dict
from copy import deepcopy

from qcfractal.interface import FractalClient

# Keywords used by QC codes
_nwc_xfine_kwds = {'values': {
    'dft__convergence__energy': '1e-8',
    'dft__grid': 'xfine',
    'basis__spherical': True
}, 'comments': 'Tight convergence settings for NWChem'}
_nwc_xfine_robust_kwds = {'values': {
    'dft__convergence__energy': '1e-8',
    'dft__grid': 'xfine',
    'dft__iterations': 250,
    'basis__spherical': True,
    'geometry__noautoz': True
}, 'comments': 'Tight convergence settings for NWChem'}
_xtb_fine_kwgs = {
    "values": {"accuracy": 0.05},
    "comments": "Tight convergence settings for XTB"
}

# Optimization specs

_opt_specs = {
    'small_basis': {
        'name': 'small_basis',
        'description': 'geomeTRIC + NWChem/B3LYP/3-21g',
        'optimization_spec': {
            'program': 'geometric',
        }, 'qc_spec': {
            'driver': 'gradient',
            'method': 'b3lyp',
            'basis': '3-21g',
            'program': 'nwchem',
            'keywords': _nwc_xfine_kwds
        }
    },
    'normal_basis': {
        'name': 'normal_basis',
        'description': 'geomeTRIC + NWCHem/B3LYP/6-31G(2df,p)',
        'optimization_spec': {
            'program': 'geometric',
        }, 'qc_spec': {
            'driver': 'gradient',
            'method': 'b3lyp',
            'basis': '6-31G(2df,p)',
            'program': 'nwchem',
            'keywords': _nwc_xfine_kwds
        }
    },
    'xtb': {
        'name': 'xtb',
        'description': 'GeomeTRIC + XTB',
        'optimization_spec': {
            'program': 'geometric',
        }, 'qc_spec': {
            "program": "xtb",
            "driver": "gradient",
            "method": "GFN2-xTB",
            "keywords": _xtb_fine_kwgs
        }
    }
}


def get_optimization_specification(client: FractalClient, name: str) -> dict:
    """Get a specification from a hard-coded list of specifications.

    Will add keywords for the specification with the QCFractal database
    so that the specification is ready to use in an OptimizationDataset

    Args:
        client: QCFractal client
        name: Name of the specification
    Returns:
        Dictionary ready to pass to QCFractal
    """

    # Lookup the desired specification
    spec = deepcopy(_opt_specs[name])

    # Add the keyword arguments for the qc_spec
    kwds = spec['qc_spec'].get('keywords', None)
    if kwds is not None:
        kwd_id = client.add_keywords([kwds])[0]
        spec['qc_spec']['keywords'] = kwd_id

    return spec


def create_computation_spec(spec_name: str, solvent: Optional[str] = None) -> dict:
    """Create a computational specification ready to use in a

    Args:
        spec_name: Name of the quantum chemistry specification
        solvent: Name of the solvent to use
    Returns:
        Specification ready to be used in a Dataset
    """

    # Lookup the specification
    output_spec: Dict[str, Dict] = deepcopy(_opt_specs[spec_name]['qc_spec'])

    # Update the keyword arguments with solvent
    kwds: dict = output_spec["keywords"]

    if solvent is not None:
        kwds["comments"] += f" in solvent {solvent}"
        program = output_spec["program"]
        if program == "xtb":
            kwds["values"]["solvent"] = solvent
        elif program == "nwchem":
            kwds["values"]["cosmo__do_cosmo_smd"] = "true"
            kwds["values"]["cosmo__solvent"] = solvent
        else:
            raise ValueError(f"Program {program} is not yet supported")
    output_spec["keywords"] = kwds

    return output_spec

