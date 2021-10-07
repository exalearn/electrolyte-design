from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json
import os

from schnetpack.data.partitioning import create_subset
from schnetpack.data import AtomsData, AtomsLoader
from schnetpack import train as trn
import schnetpack as spk
from scipy.stats import spearmanr, kendalltau
import pandas as pd
import numpy as np
from torch import optim
import torch


model_dir = Path('..')


if __name__ == "__main__":
    # Define the command line arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--batch-size', help='Number of molecules per batch', type=int, default=32)
    arg_parser.add_argument('--num-epochs', help='Number of epochs to run', type=int, default=512)
    arg_parser.add_argument('--device', default='cpu', help='Device used for training')
    arg_parser.add_argument('random_seed', help='Random seed for data bootstrap sample', type=int)

    # Parse the arguments
    args = arg_parser.parse_args()
    run_params = args.__dict__

    # Determine the output directory
    test_dir = os.path.join('networks', f'b{args.batch_size}_n{args.num_epochs}_S{args.random_seed}')
    os.makedirs(test_dir, exist_ok=True)
    with open(os.path.join(test_dir, 'config.json'), 'w') as fp:
        json.dump(run_params, fp)

    # Load in the training database and downsample it
    train_data = AtomsData('../datasets/train.db')
    sampled_idx = np.random.RandomState(args.random_seed).randint(len(train_data), size=(len(train_data),))
    sampled_idx = [int(i) for i in sampled_idx]
    train_data = create_subset(train_data, sampled_idx)

    # Making the data loaders for use during training
    train_loader = AtomsLoader(train_data, args.batch_size, shuffle=True)
    test_data = AtomsData('../datasets/test.db')
    test_loader = AtomsLoader(test_data, args.batch_size)
    valid_data = AtomsData('../datasets/valid.db')
    valid_loader = AtomsLoader(valid_data, args.batch_size)

    # Make the model
    model = torch.load('../best_model', map_location=args.device)
    for module in model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()

    # Train the model
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

    # Run on the validation set and assess statistics
    test_time = perf_counter()
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
    test_time = perf_counter() - test_time

    pd.DataFrame({'true': y_true, 'pred': y_pred}).to_csv(os.path.join(test_dir, 'test_results.csv'), index=False)

    with open(os.path.join(test_dir, 'test_summary.json'), 'w') as fp:
        json.dump({
            'r2_score': float(np.corrcoef(y_true, y_pred)[1, 0] ** 2),  # float() converts from np.float32
            'spearmanr': float(spearmanr(y_true, y_pred)[0]),
            'kendall_tau': float(kendalltau(y_true, y_pred)[0]),
            'mae': float(np.mean(np.abs(y_pred - y_true))),
            'rmse': float(np.sqrt(np.mean(np.square(y_pred - y_true))))
        }, fp, indent=2)
