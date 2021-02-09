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


class FCHLKernel(BaseEstimator):
    """Class for computing the kernel matrix using the qml utility functions

    The input `X` to all of the function is the FCHL representation vectors
    """

    def __init__(self):
        super(FCHLKernel, self).__init__()
        self.train_points = None

    def fit(self, X, y=None):
        # Store the training set
        self.train_points = np.array(X)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return np.squeeze(get_local_symmetric_kernels(self.train_points))

    def transform(self, X, y=None):
        return get_local_kernels(np.array(X), self.train_points)[0]


def evaluate_fchl(rep_computer: FCHLRepresentation, model: BaseEstimator,
                  mols: List[str], n_jobs: int = 1) -> List[float]:
    """Run an FCHL-based model

    Args:
        rep_computer: Tool used to compute the FCHL-compatible representations for each molecule
        model: Model to be evaluated
        mols: List of molecules (XYZ format) to evaluate
        n_jobs: Number of threads to use for generating representations
    Returns:
        Results from the inference
    """

    # Convert the input molecules into FCHL-ready inputs
    rep_computer.n_jobs = n_jobs
    reps = rep_computer.transform(mols)

    # Run the model
    return model.predict(reps).tolist()


def train_fchl(rep_computer: FCHLRepresentation, model: BaseEstimator,
               mols: List[str], y: List[float], n_jobs: int = 1) -> BaseEstimator:
    """Retrain an FCHL-based model

    Args:
        rep_computer: Tool used to compute the FCHL-compatible representations for each molecule
        model: Model to be retrained
        mols: List of molecules (XYZ format) in training set
        y: List of other properties to predict
        n_jobs: Number of threads to use for generating representations
    Returns:
        Retrained model
    """

    # Convert the input molecules into FCHL-ready inputs
    rep_computer.n_jobs = n_jobs
    reps = rep_computer.transform(mols)

    # Retrain the model
    return model.fit(reps, y)
