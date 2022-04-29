import hashlib
from argparse import ArgumentParser
import json
import os
from typing import List

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks as cb
from tensorflow.keras import layers
import tensorflow as tf
from scipy.stats import spearmanr, kendalltau
import pandas as pd
import numpy as np
import nfp

from moldesign.utils.callbacks import EpochTimeLogger, LRLogger
from moldesign.score.nfp import make_data_loader


def build_fn(atom_features: int = 64, message_steps: int = 8,
             output_layers: List[int] = (512, 256, 128)):
    atom = layers.Input(shape=[None], dtype=tf.int64, name='atom')
    bond = layers.Input(shape=[None], dtype=tf.int64, name='bond')
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')

    # Convert from a single integer defining the atom state to a vector
    # of weights associated with that class
    atom_state = layers.Embedding(36, atom_features, name='atom_embedding', mask_zero=True)(atom)

    # Ditto with the bond state
    bond_state = layers.Embedding(5, atom_features, name='bond_embedding', mask_zero=True)(bond)

    # Here we use our first nfp layer. This is an attention layer that looks at
    # the atom and bond states and reduces them to a single, graph-level vector.
    # mum_heads * units has to be the same dimension as the atom / bond dimension
    global_state = nfp.GlobalUpdate(units=4, num_heads=1, name='problem')([atom_state, bond_state, connectivity])

    for _ in range(message_steps):  # Do the message passing
        new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity, global_state])
        bond_state = layers.Add()([bond_state, new_bond_state])

        new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity, global_state])
        atom_state = layers.Add()([atom_state, new_atom_state])

        new_global_state = nfp.GlobalUpdate(units=4, num_heads=1)(
            [atom_state, bond_state, connectivity, global_state]
        )
        global_state = layers.Add()([global_state, new_global_state])

    # Pass the global state through an output
    output = global_state
    for shape in output_layers:
        output = layers.Dense(shape, activation='relu')(output)
    output = layers.Dense(1)(output)
    output = layers.Dense(1, activation='linear', name='scale')(output)

    # Construct the tf.keras model
    return tf.keras.Model([atom, bond, connectivity], [output])


if __name__ == "__main__":
    # Define the command line arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--atom-features', help='Number of atomic features', type=int, default=32)
    arg_parser.add_argument('--num-messages', help='Number of message-passing layers', type=int, default=8)
    arg_parser.add_argument('--output-layers', help='Number of hidden units of the output layers', type=int,
                            default=(512, 256, 128), nargs='*')
    arg_parser.add_argument('--batch-size', help='Number of molecules per batch', type=int, default=16)
    arg_parser.add_argument('--num-epochs', help='Number of epochs to run', type=int, default=64)
    arg_parser.add_argument('--padded-size', help='Maximum number of atoms per molecule', type=int, default=None)

    # Parse the arguments
    args = arg_parser.parse_args()
    run_params = args.__dict__
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]

    # Determine the output directory
    test_dir = os.path.join('networks', f'T{args.num_messages}_b{args.batch_size}_n{args.num_epochs}_{params_hash}')
    os.makedirs(test_dir)
    with open(os.path.join(test_dir, 'config.json'), 'w') as fp:
        json.dump(run_params, fp)

    # Making the data loaders
    train_data = pd.read_csv('data/train.csv')
    train_loader = make_data_loader(train_data['smiles'], train_data['output'], shuffle_buffer=32768,
                                    batch_size=args.batch_size, max_size=args.padded_size, drop_last_batch=True)

    test_data = pd.read_csv('data/test.csv')
    test_loader = make_data_loader(test_data['smiles'], test_data['output'], batch_size=args.batch_size,
                                   max_size=args.padded_size, drop_last_batch=True)

    valid_data = pd.read_csv('data/valid.csv')
    valid_loader = make_data_loader(valid_data['smiles'], valid_data['output'], batch_size=args.batch_size,
                                    max_size=args.padded_size, drop_last_batch=True)

    # Make the model
    model = build_fn(atom_features=args.atom_features, message_steps=args.num_messages,
                     output_layers=args.output_layers)

    # Set the scale for the output parameter
    ic50s = np.concatenate([x[1].numpy() for x in iter(train_loader)], axis=0)
    model.get_layer('scale').set_weights([np.array([[ic50s.std()]]), np.array([ic50s.mean()])])

    # Train the model
    final_learn_rate = 1e-6
    init_learn_rate = 1e-3
    decay_rate = (final_learn_rate / init_learn_rate) ** (1. / (args.num_epochs - 1))


    def lr_schedule(epoch, lr):
        return lr * decay_rate


    model.compile(Adam(init_learn_rate), 'mean_squared_error', metrics=['mean_absolute_error'])
    history = model.fit(
        train_loader, validation_data=valid_loader, epochs=args.num_epochs, verbose=True,
        shuffle=False, callbacks=[
            LRLogger(),
            EpochTimeLogger(),
            cb.LearningRateScheduler(lr_schedule),
            cb.ModelCheckpoint(os.path.join(test_dir, 'best_model.h5'), save_best_only=True),
            cb.EarlyStopping(patience=128, restore_best_weights=True),
            cb.CSVLogger(os.path.join(test_dir, 'train_log.csv')),
            cb.TerminateOnNaN()
        ]
    )

    # Run on the validation set and assess statistics
    y_true = np.hstack([np.squeeze(x[1].numpy()) for x in iter(test_loader)])
    y_pred = np.squeeze(model.predict(test_loader))

    pd.DataFrame({'true': y_true, 'pred': y_pred}).to_csv(os.path.join(test_dir, 'test_results.csv'), index=False)

    with open(os.path.join(test_dir, 'test_summary.json'), 'w') as fp:
        json.dump({
            'r2_score': float(np.corrcoef(y_true, y_pred)[1, 0] ** 2),  # float() converts from np.float32
            'spearmanr': float(spearmanr(y_true, y_pred)[0]),
            'kendall_tau': float(kendalltau(y_true, y_pred)[0]),
            'mae': float(np.mean(np.abs(y_pred - y_true))),
            'rmse': float(np.sqrt(np.mean(np.square(y_pred - y_true))))
        }, fp, indent=2)
