from time import perf_counter

import numpy as np

from moldesign.score.mpnn import MPNNMessage, update_mpnn, evaluate_mpnn, retrain_mpnn


def test_train(train_dataset, model):
    # Make the MPNN into a message object
    model_msg = MPNNMessage(model)
    new_weights, history = update_mpnn(model_msg, train_dataset, 2, validation_split=0.5)
    assert 'val_loss' in history
    assert len(new_weights) == len(model_msg.weights)

    # Try training from fresh
    new_weights, history = retrain_mpnn(model.get_config(), train_dataset,
                                        2, validation_split=0.5)
    assert 'val_loss' in history
    assert len(new_weights) == len(model_msg.weights)

    # Try training from fresh, with bootstrap
    new_weights, history = retrain_mpnn(model.get_config(), train_dataset,
                                        2, validation_split=0.5, bootstrap=True)
    assert 'val_loss' in history
    assert len(new_weights) == len(model_msg.weights)

    # Test with a timeout
    start_time = perf_counter()
    update_mpnn(model_msg, train_dataset, 512, validation_split=0.5, timeout=1)
    assert perf_counter() - start_time < 2

    # Test with a test set
    _, _, y_pred = \
        update_mpnn(model_msg, train_dataset, 512, ['C', 'CC'],
                    validation_split=0.5, timeout=1)
    assert np.array(y_pred).shape == (2,)


def test_inference(model):
    # Evaluate serial, then it parallel
    results_serial = evaluate_mpnn([model], ['C', 'CC'])
    results_parallel = evaluate_mpnn([model], ['C', 'CC'], n_jobs=2)
    assert np.isclose(results_parallel, results_serial).all()
