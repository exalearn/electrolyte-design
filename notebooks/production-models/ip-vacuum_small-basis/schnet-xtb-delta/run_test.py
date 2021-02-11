import hashlib
from argparse import ArgumentParser
import json
import os
from typing import Optional

import schnetpack as spk
from schnetpack import AtomsData, AtomsLoader
from schnetpack import train as trn
from torch import optim
import torch
from scipy.stats import spearmanr, kendalltau
import pandas as pd
import numpy as np


def build_fn(atom_features: int = 128, message_steps: int = 8,
             output_layers: int = 3, reduce_fn: str = 'sum',
             mean: Optional[float] = None, std: Optional[float] = None):
    schnet = spk.representation.SchNet(
        n_atom_basis=atom_features, n_filters=atom_features,
        n_gaussians=20, n_interactions=message_steps,
        cutoff=4., cutoff_network=spk.nn.cutoff.CosineCutoff
    )

    output = spk.atomistic.Atomwise(n_in=atom_features, n_layers=output_layers, aggregation_mode=reduce_fn, mean=mean, stddev=std,
                                    property='delta')
    return spk.AtomisticModel(representation=schnet, output_modules=output)


if __name__ == "__main__":
    # Define the command line arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--atom-features', help='Number of atomic features', type=int, default=128)
    arg_parser.add_argument('--num-messages', help='Number of message-passing layers', type=int, default=3)
    arg_parser.add_argument('--output-layers', help='Number of hidden units of output layers', type=int, default=3)
    arg_parser.add_argument('--batch-size', help='Number of molecules per batch', type=int, default=32)
    arg_parser.add_argument('--num-epochs', help='Number of epochs to run', type=int, default=64)
    arg_parser.add_argument('--readout-fn', default='sum', help='Readout function')
    arg_parser.add_argument('--device', default='cpu', help='Device used for training')

    # Parse the arguments
    args = arg_parser.parse_args()
    run_params = args.__dict__
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]

    # Determine the output directory
    test_dir = os.path.join('networks', f'T{args.num_messages}_b{args.batch_size}_n{args.num_epochs}_{params_hash}')
    os.makedirs(test_dir, exist_ok=True)  # TODO
    with open(os.path.join(test_dir, 'config.json'), 'w') as fp:
        json.dump(run_params, fp)

    # Making the data loaders
    train_data = AtomsData('datasets/train.db')
    train_loader = AtomsLoader(train_data, args.batch_size, shuffle=True)
    test_data = AtomsData('datasets/test.db')
    test_loader = AtomsLoader(test_data, args.batch_size)
    valid_data = AtomsData('datasets/valid.db')
    valid_loader = AtomsLoader(valid_data, args.batch_size)

    # Make the model
    mean, std = train_loader.get_statistics('delta', divide_by_atoms=True)
    model = build_fn(atom_features=args.atom_features, message_steps=args.num_messages,
                     output_layers=args.output_layers, reduce_fn=args.readout_fn,
                     mean=mean['delta'], std=std['delta'])

    # Train the model
    #  Following:
    init_learn_rate = 1e-4
    opt = optim.Adam(model.parameters(), lr=init_learn_rate)

    loss = trn.build_mse_loss(['delta'])
    metrics = [spk.metrics.MeanSquaredError('delta')]
    hooks = [
        trn.CSVHook(log_path=test_dir, metrics=metrics),
        trn.ReduceLROnPlateauHook(
            opt,
            patience=args.num_epochs // 8, factor=0.8, min_lr=1e-6,
            stop_after_min=True
        )
    ]

    trainer = trn.Trainer(
        model_path=test_dir,
        model=model,
        hooks=hooks,
        loss_fn=loss,
        optimizer=opt,
        train_loader=train_loader,
        validation_loader=valid_loader,
    )

    trainer.train(args.device, n_epochs=args.num_epochs)

    # Load in the best model
    model = torch.load(os.path.join(test_dir, 'best_model'))

    # Run on the validation set and assess statistics
    y_true = []
    y_pred = []
    for batch in test_loader:
        batch = {k: v.to(args.device) for k, v in batch.items()}

        # apply model
        pred = model(batch)
        y_pred.append(pred['delta'].detach().cpu().numpy())
        y_true.append(batch['delta'].detach().cpu().numpy())

    y_true = np.squeeze(np.concatenate(y_true))
    y_pred = np.squeeze(np.concatenate(y_pred))

    pd.DataFrame({'true': y_true, 'pred': y_pred}).to_csv(os.path.join(test_dir, 'test_results.csv'), index=False)

    with open(os.path.join(test_dir, 'test_summary.json'), 'w') as fp:
        json.dump({
            'r2_score': float(np.corrcoef(y_true, y_pred)[1, 0] ** 2),  # float() converts from np.float32
            'spearmanr': float(spearmanr(y_true, y_pred)[0]),
            'kendall_tau': float(kendalltau(y_true, y_pred)[0]),
            'mae': float(np.mean(np.abs(y_pred - y_true))),
            'rmse': float(np.sqrt(np.mean(np.square(y_pred - y_true))))
        }, fp, indent=2)
