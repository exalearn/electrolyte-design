from moldesign.sample.rl.envs.simple import Molecule
from moldesign.utils.conversions import convert_smiles_to_nx


def test_reward():
    env = Molecule()
    assert env._state is None
    assert env.reward() == 0

    env = Molecule(init_mol=convert_smiles_to_nx('C'))
    env.reward()
