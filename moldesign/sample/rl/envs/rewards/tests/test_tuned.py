from moldesign.sample.rl.envs.rewards.mpnn import MPNNReward
from moldesign.sample.rl.envs.rewards.oneshot import OneShotScore
from moldesign.sample.rl.envs.rewards.tuned import LogisticCombination
from moldesign.utils.conversions import convert_smiles_to_nx


def test_reward(model, oneshot_model, atom_types, bond_types, target_mols):
    ic50_reward = MPNNReward(model, atom_types, bond_types)
    sim_reward = OneShotScore(oneshot_model, atom_types, bond_types, target_mols)
    reward = LogisticCombination(ic50_reward, sim_reward)

    graph = convert_smiles_to_nx('C')
    x = reward(graph)
    assert isinstance(x, float)
    assert 0 < x < 3
