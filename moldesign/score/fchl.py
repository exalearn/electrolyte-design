"""Tools for using the FCHL representation"""
from io import StringIO
from typing import List

import numpy as np
from qml.fchl import get_local_kernels, get_local_symmetric_kernels
from qml import Compound
from sklearn.base import BaseEstimator

from moldesign.utils.globals import get_process_pool


def _compute_representation(xyz: str, max_size: int = 75) -> List[np.ndarray]:
    """Compute the representation for a molecule"""
    c = Compound(StringIO(xyz))
    c.generate_fchl_representation(max_size=max_size)
    return c.representation


class FCHLRepresentation(BaseEstimator):
    """Converts molecules from XYZ to FCHL representation format"""

    def __init__(self, max_size: int = 75, n_jobs: int = 1):
        """
        Args:
            max_size: Maximum size of the input molecule
            n_jobs: Number of processes used to compute the formation enthalpy
        """
        self.max_size = max_size
        self.n_jobs = n_jobs

    def transform(self, X, y=None):
        if self.n_jobs == 1:
            return np.array([_compute_representation(x, self.max_size) for x in X])
        else:
            pool = get_process_pool(self.n_jobs)
            return np.array(pool.map(_compute_representation, X))


def _compute_average(kernel: np.array, reps_i: List[np.array], reps_j: List[np.array]):
    """Compute the average kernel

    Args:
        kernel: Kernel to be averaged
        reps_i: Representations for the "rows" of the kernel matrix
        reps_j: Representations for the "columns" of the kernel matrix
    """

    # Count the number of atoms in the rows and columns
    #  Works by accessing where the atomic number is stored in the FCHL representation
    natoms_i = np.array([np.greater(x[:][0][1], 0).sum() for x in reps_i])
    natoms_j = np.array([np.greater(x[:][0][1], 0).sum() for x in reps_j])
    total_atoms = natoms_i[:, None] * natoms_j[None, :]

    # Compute the average
    kernel /= total_atoms


class FCHLKernel(BaseEstimator):
    """Class for computing the kernel matrix using the qml utility functions

    The input `X` to all of the function is the FCHL representation vectors
    """

    def __init__(self, average: bool = False):
        """
        Args:
            average: Whether to compute the average kernel
        """
        super(FCHLKernel, self).__init__()
        self.average = average
        self.train_points = None

    def fit(self, X, y=None):
        # Store the training set
        self.train_points = np.array(X)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        # Compute the kernel matrix by only computing the lower half
        kernel = get_local_symmetric_kernels(self.train_points)[0]

        # If specified, average it
        if self.average:
            _compute_average(kernel, X, X)
        return kernel

    def transform(self, X, y=None):
        # Compute the kernel matrix
        kernel = get_local_kernels(np.array(X), self.train_points)[0]
        if self.average:
            _compute_average(kernel, X, self.train_points)
        return kernel


def evaluate_fchl(rep_computer: FCHLRepresentation, model: BaseEstimator,
                  mols: List[str], n_jobs: int = 1, y_lower: List[float] = None) -> np.ndarray:
    """Run an FCHL-based model

    Args:
        rep_computer: Tool used to compute the FCHL-compatible representations for each molecule
        model: Model to be evaluated
        mols: List of molecules (XYZ format) to evaluate
        n_jobs: Number of threads to use for generating representations
        y_lower: Lower-fidelity estimate of the property. Used for delta learning models
    Returns:
        Results from the inference
    """

    # Convert the input molecules into FCHL-ready inputs
    rep_computer.n_jobs = n_jobs
    reps = rep_computer.transform(mols)

    # Run the model
    y_pred = model.predict(reps).tolist()
    if y_lower is not None:
        y_pred = np.add(y_pred, y_lower)
    return y_pred


def train_fchl(rep_computer: FCHLRepresentation, model: BaseEstimator,
               mols: List[str], y: List[float], n_jobs: int = 1, y_lower: List[float] = None) -> BaseEstimator:
    """Retrain an FCHL-based model

    Args:
        rep_computer: Tool used to compute the FCHL-compatible representations for each molecule
        model: Model to be retrained
        mols: List of molecules (XYZ format) in training set
        y: List of other properties to predict
        n_jobs: Number of threads to use for generating representations
        y_lower: Lower-fidelity estimate of the property. Used for delta learning models
    Returns:
        Retrained model
    """

    # Convert the input molecules into FCHL-ready inputs
    rep_computer.n_jobs = n_jobs
    reps = rep_computer.transform(mols)

    # Retrain the model
    if y_lower is not None:
        y = np.subtract(y, y_lower)
    return model.fit(reps, y)
