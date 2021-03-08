"""Thermodynamic functions"""

import logging
from typing import List, Union, Dict

from qcelemental.models import Molecule
from qcelemental.physical_constants import constants
from rdkit import Chem
import numpy as np

logger = logging.getLogger(__name__)

c = constants.ureg.Quantity(constants.c, constants.get('c', True).units)
h = constants.ureg.Quantity(constants.h, constants.get('h', True).units)
kb = constants.ureg.Quantity(constants.kb, constants.get('kb', True).units)
r = constants.ureg.Quantity(constants.R, constants.get('R', True).units)


def subtract_reference_energies(total_energy: float, mol: Union[Molecule, Chem.Mol],
                                reference_energies: Dict[str, float]) -> float:
    """Compute the atomization energy by subtracting off reference energies

    Args:
        total_energy (float): Total energy of a molecule
        mol: Molecule of interest
        reference_energies (float): Isolated atom energies in the same units as total_energy
    Returns:
        (float) Atomization energy
    """
    # Get the elements
    if isinstance(mol, Molecule):
        symbols = mol.symbols
    elif isinstance(mol, Chem.Mol):
        symbols = [x.GetSymbol() for x in mol.GetAtoms()]
    else:
        raise ValueError(f'Unrecognized format: {type(mol)}')

    # Subtract off the reference energies
    atom_energy = total_energy
    for label in symbols:
        atom_energy -= reference_energies[label]

    # Get the output energy
    return atom_energy


def mass_weighted_hessian(hessian: np.ndarray, molecule: Molecule):
    """Compute the mass-weighted hessian

    Args:
        hessian: Hessian for a molecule
        molecule (Molecule): Molecule used for the Hessian calculation
    Returns:
        Weighted hessian
    """
    # Make a copy of the Hessian
    output = hessian.copy()

    # Apply masses
    masses = molecule.masses
    for i in range(len(masses)):
        for j in range(len(masses)):
            output[3 * i:3 * i + 3, 3 * j:3 * j + 3] /= np.sqrt(masses[i] * masses[j])
    return output


def compute_frequencies(hessian: np.ndarray, molecule: Molecule,
                        units: str = 'hartree / bohr ** 2') -> np.array:
    """Compute the characteristic temperature of vibrational frequencies for a molecule

    Args:
        hessian: Hessian matrix
        molecule: Molecule object
        units: Units for the Hessian matrix
    Returns:
        ([float]): List of vibrational frequencies in Hz
    """

    # Compute the mass-weighted hessian
    mass_hessian = mass_weighted_hessian(hessian, molecule)

    # Compute the eigenvalues and compute frequencies
    eig = np.linalg.eigvals(mass_hessian)
    freq = np.sign(eig) * np.sqrt(np.abs(eig))
    conv = np.sqrt(constants.conversion_factor(f'{units} / amu', 'Hz ** 2'))
    freq *= conv / np.pi / 2  # Converts from angular to ordinary frequency
    return freq


def compute_wavenumbers(hessian: np.ndarray, molecule: Molecule,
                        units: str = 'hartree / bohr ** 2') -> np.array:
    """Compute the wavenumbers for vibrations in the molecule

    Args:
        hessian: Hessian matrix
        molecule: Molecule object
        units: Units for the Hessian matrix
    Returns:
        ([float]): List of vibrational wavenumbers in 1/cm"""

    # Compute the vibrational frequencies
    freq = compute_frequencies(hessian, molecule, units)
    freq = constants.ureg.Quantity(freq, 'Hz')

    return (freq / c).to("1 / cm").magnitude


def compute_zpe(hessian: np.ndarray, molecule: Molecule,
                scaling: float = 1, units: str = 'hartree / bohr ** 2',
                verbose: bool = False) -> float:
    """Compute the zero-point energy of a molecule

    Args:
        hessian: Hessian matrix
        molecule: Molecule object
        scaling: How much to scale frequencies before computing ZPE
        units: Units for the Hessian matrix
        verbose: Whether to display warnings about negative frequencies
    Returns:
        (float) Energy for the system in Hartree
    """

    # Get the characteristic temperatures of all vibrations
    freqs = compute_frequencies(hessian, molecule, units)
    freqs = constants.ureg.Quantity(freqs, 'Hz')

    # Get the name of the molecule
    name = molecule.name

    return compute_zpe_from_freqs(freqs, scaling, verbose, name)


def compute_zpe_from_freqs(freqs: List[float], scaling: float = 1, verbose: bool = False, name: str = "Molecule"):
    """Compute the zero-point energy of a molecule

    Args:
        freqs: Frequencies in Hz
        scaling: How much to scale frequencies before computing ZPE
        verbose: Whether to display warnings about negative frequencies
        name: Name of the molecule, used when displaying warnings
    Returns:
        (float) Energy for the system in Hartree
    """

    # Make sure they are an ndarray
    freqs = constants.ureg.Quantity(freqs, "Hz")

    # Scale them
    freqs *= scaling

    # Drop the negative frequencies
    neg_freqs = freqs[freqs < 0]
    if len(neg_freqs) > 0:
        wavenumbers = neg_freqs / c

        # Remove those with a wavenumber less than 80 cm^-1 (basically zero)
        wavenumbers = wavenumbers[wavenumbers.to("1/cm").magnitude < -80]
        if len(wavenumbers) > 0:
            output = ' '.join(f'{x:.2f}' for x in wavenumbers.to("1/cm").magnitude)
            if verbose:
                logger.warning(f'{name} has {len(neg_freqs)} negative components. Largest: [{output}] cm^-1')

    #  Convert the frequencies to characteristic temps
    freqs = constants.ureg.Quantity(freqs, 'Hz')
    temps = (h * freqs / kb)

    # Filter out temperatures less than 300 K (using this as a threshold for negative modes
    temps = temps[np.array(temps.to("K")) > 300.]

    # Compute the ZPE
    zpe = r * temps.sum() / 2
    return zpe.to('hartree').magnitude
