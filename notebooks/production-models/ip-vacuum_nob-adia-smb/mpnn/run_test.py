import hashlib
from time import perf_counter
from argparse import ArgumentParser
import json
import os
from typing import List
from pathlib import Path

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks as cb
from scipy.stats import spearmanr, kendalltau
import pandas as pd
import numpy as np

from moldesign.score.mpnn.callbacks import EpochTimeLogger, LRLogger
from moldesign.score.mpnn.layers import GraphNetwork, Squeeze
from moldesign.score.mpnn.data import make_data_loader

_data_dir = Path('../datasets')
output = 'output'


def build_fn(atom_features: int = 64, message_steps: int = 8,
             output_layers: List[int] = (512, 256, 128), reduce_fn: str = 'softmax',
             atomic_contribution: bool = False):
    node_graph_indices = Input(shape=(1,), name='node_graph_indices', dtype='int32')
    atom_types = Input(shape=(1,), name='atom', dtype='int32')
    bond_types = Input(shape=(1,), name='bond', dtype='int32')
    connectivity = Input(shape=(2,), name='connectivity', dtype='int32')

    # Squeeze the node graph and connectivity matrices
    snode_graph_indices = Squeeze(axis=1)(node_graph_indices)
    satom_types = Squeeze(axis=1)(atom_types)
    sbond_types = Squeeze(axis=1)(bond_types)

    output = GraphNetwork(atom_type_count, bond_type_count, atom_features, message_steps,
                          output_layer_sizes=output_layers,
                          atomic_contribution=atomic_contribution, reduce_function=reduce_fn,
                          name='mpnn')([satom_types, sbond_types, snode_graph_indices, connectivity])

    # Scale the output
    output = Dense(1, activation='linear', name='scale')(output)

    return Model(inputs=[node_graph_indices, atom_types, bond_types, connectivity],
                 outputs=output)


if __name__ == "__main__":
    # Define the command line arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--atom-features', help='Number of atomic features', type=int, default=128)
    arg_parser.add_argument('--num-messages', help='Number of message-passing layers', type=int, default=8)
    arg_parser.add_argument('--output-layers', help='Number of hidden units of the output layers', type=int,
                            default=(512, 256, 128), nargs='*')
    arg_parser.add_argument('--batch-size', help='Number of molecules per batch', type=int, default=32)
    arg_parser.add_argument('--num-epochs', help='Number of epochs to run', type=int, default=64)
    arg_parser.add_argument('--readout-fn', default='softmax', help='Readout function')
    arg_parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing runs.')
    arg_parser.add_argument('--atomwise', action='store_true', help='Use an atomic contribution network')

    # Parse the arguments
    args = arg_parser.parse_args()
    run_params = args.__dict__
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]

    # Determine the output directory
    test_dir = os.path.join('networks', f'T{args.num_messages}_b{args.batch_size}_n{args.num_epochs}_{params_hash}')
    os.makedirs(test_dir, exist_ok=args.overwrite)
    with open(os.path.join(test_dir, 'config.json'), 'w') as fp:
        json.dump(run_params, fp)

    # Making the data loaders
    train_loader = make_data_loader(str(_data_dir / 'train_data.proto'), shuffle_buffer=32768, cache=True,
                                    batch_size=args.batch_size, output_property=output)
    test_loader = make_data_loader(str(_data_dir / 'test_data.proto'),
                                   batch_size=args.batch_size, output_property=output)
    val_loader = make_data_loader(str(_data_dir / 'valid_data.proto'),
                                  batch_size=args.batch_size, output_property=output, cache=True)

    # Load in the bond and atom type information
    with open(_data_dir / '../atom_types.json') as fp:
        atom_type_count = len(json.load(fp))
    with open(_data_dir / '../bond_types.json') as fp:
        bond_type_count = len(json.load(fp))

    # Make the model
    model = build_fn(atom_features=args.atom_features, message_steps=args.num_messages,
                     output_layers=args.output_layers, reduce_fn=args.readout_fn,
                     atomic_contribution=args.atomwise)

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
        train_loader, validation_data=val_loader, epochs=args.num_epochs, verbose=True,
        shuffle=False, callbacks=[
            LRLogger(),
            EpochTimeLogger(),
            cb.LearningRateScheduler(lr_schedule),
            cb.ModelCheckpoint(os.path.join(test_dir, 'best_model.h5'), save_best_only=True),
            cb.EarlyStopping(patience=args.num_epochs // 8, restore_best_weights=True),
            cb.CSVLogger(os.path.join(test_dir, 'train_log.csv')),
            cb.TerminateOnNaN()
        ]
    )

    # Run on the validation set and assess statistics
    y_true = np.hstack([np.squeeze(x[1].numpy()) for x in iter(test_loader)])
    test_time = perf_counter()
    y_pred = np.squeeze(model.predict(test_loader))
    test_time = perf_counter() - test_time

    pd.DataFrame({'true': y_true, 'pred': y_pred}).to_csv(os.path.join(test_dir, 'test_results.csv'), index=False)

    with open(os.path.join(test_dir, 'test_summary.json'), 'w') as fp:
        json.dump({
            'r2_score': float(np.corrcoef(y_true, y_pred)[1, 0] ** 2),  # float() converts from np.float32
            'spearmanr': float(spearmanr(y_true, y_pred)[0]),
            'kendall_tau': float(kendalltau(y_true, y_pred)[0]),
            'mae': float(np.mean(np.abs(y_pred - y_true))),
            'rmse': float(np.sqrt(np.mean(np.square(y_pred - y_true)))),
            'test_time': test_time
        }, fp, indent=2)
