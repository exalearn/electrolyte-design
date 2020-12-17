from moldesign.score.mpnn import MPNNMessage, update_mpnn


def test_train(train_dataset, model):
    # Make the MPNN into a message object
    model_msg = MPNNMessage(model)
    new_weights, history = update_mpnn(model_msg, train_dataset,
                                       2, [1, 6], ['SINGLE'], validation_split=0.5)
    assert 'val_loss' in history
    assert len(new_weights) == len(model_msg.weights)
