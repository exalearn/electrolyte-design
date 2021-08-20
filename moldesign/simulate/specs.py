"""Specifications for different levels of computational chemistry"""
from typing import Optional, Dict, Tuple, Union
from copy import deepcopy

from qcelemental.models import AtomicResult, OptimizationResult
from qcelemental.models.common_models import Model
from qcelemental.models.procedures import QCInputSpecification
from qcfractal.interface import FractalClient

# Keywords used by QC codes
from qcfractal.interface.models import OptimizationRecord, ResultRecord

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
    'diffuse_basis': {
        'name': 'diffuse_basis',
        'description': 'geomeTRIC + NWCHem/B3LYP/6-311++gss',
        'optimization_spec': {
            'program': 'geometric',
        }, 'qc_spec': {
            'driver': 'gradient',
            'method': 'b3lyp',
            'basis': '6-311++G**',
            'program': 'nwchem',
            'keywords': _nwc_xfine_robust_kwds
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

# Reference energies for each configuration
_reference_energies = {
    'xtb': {'H': -0.3934827639359724,
            'He': -1.743126632945867,
            'Li': -0.18007168657517492,
            'C': -1.7932963713649235,
            'N': -2.6058241612788278,
            'O': -3.767606950375682,
            'F': -4.619339964237827,
            'Si': -1.569609938455468,
            'P': -2.3741787947323725,
            'S': -3.146456870402072,
            'Cl': -4.4825251349610635,
            'Br': -4.048339371234208,
            'I': -3.7796302633896515},
    'small_basis': {'H': -0.497311388804,
                    'He': -2.886001303629,
                    'Li': -7.438943611544,
                    'C': -37.64269644992,
                    'N': -54.295462727225,
                    'O': -74.660293277123,
                    'F': -99.182166194876,
                    'Si': -287.866879627857,
                    'P': -339.548419942544,
                    'S': -396.162245759273,
                    'Cl': -457.945732528969,
                    'Br': -2561.754609523183,
                    'I': -6889.992449675247},
    'normal_basis': {'H': -0.500272782422,
                     'He': -2.9070481031,
                     'Li': -7.490902306945,
                     'C': -37.844958497185,
                     'N': -54.582875607216,
                     'O': -75.060582294288,
                     'F': -99.715958130901,
                     'Si': -289.370112438377,
                     'P': -341.255344529106,
                     'S': -398.103353899211,
                     'Cl': -460.134289124795},
    'diffuse_basis': {'H': -0.502256981381,
                      'He': -2.913027858378,
                      'Li': -7.491333092405,
                      'C': -37.857266645766,
                      'N': -54.600723364844,
                      'O': -75.089879446124,
                      'F': -99.760580091674,
                      'Si': -289.394168463041,
                      'P': -341.281731880244,
                      'S': -398.133069981595,
                      'Cl': -460.166882319441},
    'g4mp2': {'C': -37.794203,
              'H': -0.502094,
              'O': -75.002483,
              'N': -54.532825,
              'Li': -7.434837,
              'F': -99.659686}}

levels = list(_reference_energies.keys())


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


def get_computation_specification(spec_name: str, solvent: Optional[str] = None) -> dict:
    """Create a computational specification ready to use in a QCFractal dataset

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


def get_qcinput_specification(spec_name: str, solvent: Optional[str] = None) -> Tuple[QCInputSpecification, str]:
    """Get the computational specification in a QCEngine-ready format

    Args:
        spec_name: Name of the quantum chemistry specification
        solvent: Name of the solvent to use
    Returns:
        - Input specification
        - Name of the program
    """

    # Make the specification
    spec = get_computation_specification(spec_name, solvent)

    # Reshape it to QCInputSpecification
    program = spec.pop("program")
    spec["model"] = {
        "method": spec.pop("method"),
    }
    if "basis" in spec:
        spec["model"]["basis"] = spec.pop("basis")
    if "keywords" in spec:
        spec["keywords"] = spec["keywords"]["values"]

    return QCInputSpecification(**spec), program


def lookup_reference_energies(spec_name: str) -> Dict[str, float]:
    """Get the atomic reference energies for a certain specification

    Args:
        spec_name: Name of the quantum chemistry specification
    Returns:
        Map of element name to reference energy
    """

    return _reference_energies[spec_name]


def infer_specification_from_result(result: Union[AtomicResult, ResultRecord,
                                                  OptimizationResult, OptimizationRecord],
                                    client: Optional[FractalClient]) -> Tuple[str, Optional[str]]:
    """Infer the specification used to run a simulation given the result file

    Args:
        result: A QCEngine result
        client: Needed to get the keywords from a OptimizationRecord or ResultRecord
    Returns:
        - The specification used to produce this result
        - Name of the solvent, if any
    """

    # Extract the method and keywords
    if isinstance(result, AtomicResult):
        model = result.model
        kwds = AtomicResult.keywords
    elif isinstance(result, OptimizationResult):
        model = result.input_specification.model
        kwds = result.input_specification.keywords
    elif isinstance(result, OptimizationRecord):
        model = Model(method=result.qc_spec.method, basis=result.qc_spec.basis)
        assert client is not None, "You must provide the client along with OptimizationRecords"
        kwds_id = result.qc_spec.keywords
        kwds = client.query_keywords(kwds_id)[0].values
    elif isinstance(result, ResultRecord):
        model = result
        assert client is not None, "You must provide the client along with ResultRecords"
        kwds_id = model.keywords
        kwds = client.query_keywords(kwds_id)[0].values
    else:
        raise ValueError(f'Unrecognized object type: {type(result)}')

    # Match to the specifications
    for name, spec in _opt_specs.items():
        qc_spec = spec["qc_spec"]
        if qc_spec["method"].lower() == model.method.lower() and \
                model.dict(include={'basis'}, exclude_none=True).get("basis", "") \
                == qc_spec.get("basis", "").lower():

            # Pull the solvent from the keywords
            if qc_spec['program'] == 'xtb':
                solvent = kwds.get('solvent')
            elif qc_spec['program'] == 'nwchem':
                solvent = kwds.get("cosmo__solvent")
            else:
                raise ValueError(f'We have not implemented how to get solvent from {spec["program"]}')
            return name, solvent
    raise KeyError(f"Method not matched. method={model.method} basis={model.basis}")
