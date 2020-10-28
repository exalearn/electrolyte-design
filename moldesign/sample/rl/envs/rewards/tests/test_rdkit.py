"""Tests for the RDKit rewards"""

from moldesign.sample.rl.envs.rewards.rdkit import QEDReward, SAScore, LogP, CycleLength
from moldesign.utils.conversions import convert_smiles_to_nx


def test_all():
    graph = convert_smiles_to_nx('CC')
    assert isinstance(QEDReward()(graph), float)
    assert isinstance(LogP()(graph), float)
    assert isinstance(SAScore()(graph), float)
    assert isinstance(CycleLength()(graph), float)
