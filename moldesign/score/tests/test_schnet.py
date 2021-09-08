import pickle as pkl
from pathlib import Path
from typing import List

import torch
import numpy as np
from pytest import fixture

from moldesign.score.schnet import TorchMessage, evaluate_schnet, AddToRepresentation
from moldesign.simulate.functions import generate_inchi_and_xyz

_model_file = Path(__file__).parent.joinpath('best_model')


@fixture()
def model() -> torch.nn.Module:
    return torch.load(_model_file, map_location='cpu')


@fixture()
def molecules() -> List[str]:
    return [generate_inchi_and_xyz(x)[1] for x in ['C', 'CC', 'O']]


def test_message(model):
    msg = TorchMessage(model)
    msg_bytes = pkl.dumps(msg)
    msg_2 = pkl.loads(msg_bytes)
    model_2 = msg_2.get_model()
    assert isinstance(model_2, torch.nn.Module)


def test_inference(model, molecules):
    y_pred = evaluate_schnet([model, model], molecules, 'delta')
    assert y_pred.shape == (3, 2)
    assert np.max(y_pred[:, 0] - y_pred[:, 1]) == 0


def test_add_features():
    l = AddToRepresentation(['test'])

    # Test with atomic level features
    output = l({
        'representation': torch.zeros((4, 8, 4)),
        'test': torch.unsqueeze(torch.arange(8), 0).expand(4, -1)
    })
    assert (output[:, :, -1] == torch.arange(8)).all()

    # Test with molecule-level features
    output = l({
        'representation': torch.zeros((4, 8, 4)),
        'test': torch.arange(4)
    })
    assert (output[:, 0, -1] == torch.arange(4)).all()
