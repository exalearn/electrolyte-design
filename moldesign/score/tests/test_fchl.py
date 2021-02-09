from typing import Tuple, List

import numpy as np
from pytest import fixture, mark
from sklearn.base import BaseEstimator
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline

from moldesign.score.fchl import FCHLRepresentation, FCHLKernel, train_fchl, evaluate_fchl
from moldesign.simulate.functions import generate_inchi_and_xyz


@fixture
def training_set() -> Tuple[List[str], List[float]]:
    xyzs = [generate_inchi_and_xyz(x)[1] for x in ['C', 'CC']]
    return xyzs, [1., 2.]


@fixture
def model() -> Tuple[FCHLRepresentation, BaseEstimator]:
    model = Pipeline([
        ('kernel', FCHLKernel()),
        ('model', KernelRidge(alpha=1e-7))
    ])
    return FCHLRepresentation(n_jobs=1), model


@mark.parametrize('n_jobs', [1, 2])
def test_train_and_run(model, training_set, n_jobs):
    # Unpack the inputs
    rep, model = model
    xyzs, ys = training_set

    # Fit the model
    model = train_fchl(rep, model, xyzs, ys, n_jobs=n_jobs)
    preds = evaluate_fchl(rep, model, xyzs, n_jobs=n_jobs)
    assert np.isclose(preds, ys).all()
