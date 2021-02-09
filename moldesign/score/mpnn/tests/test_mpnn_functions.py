import numpy as np

from moldesign.score.mpnn import MPNNMessage, update_mpnn, evaluate_mpnn


def test_train(train_dataset, model, atom_types, bond_types):
    # Make the MPNN into a message object
    model_msg = MPNNMessage(model)
    new_weights, history = update_mpnn(model_msg, train_dataset,
                                       2, atom_types, bond_types, validation_split=0.5)
    assert 'val_loss' in history
    assert len(new_weights) == len(model_msg.weights)


def test_inference(model, atom_types, bond_types):
    # Evaluate serial, then it parallel
    results_serial = evaluate_mpnn([model], ['C', 'CC'], atom_types, bond_types)
    results_parallel = evaluate_mpnn([model], ['C', 'CC'], atom_types, bond_types, n_jobs=2)
    assert np.isclose(results_parallel, results_serial).all()
