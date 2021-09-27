"""Utilities for using models based on SchNet"""
from time import monotonic
from typing import Union, List, Optional, Dict, Tuple
from tempfile import TemporaryDirectory
from io import BytesIO, StringIO
from pathlib import Path
import os

import pandas as pd
from schnetpack.data import AtomsData, AtomsLoader
from schnetpack import train as trn
from schnetpack.atomistic import Atomwise
from schnetpack import Properties
import schnetpack as spk
from ase.io.xyz import read_xyz
from torch.autograd import grad
from torch import optim
from torch import nn
import numpy as np
import torch


class TorchMessage:
    """Send a PyTorch object via pickle, enable loading on to target hardware"""

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model: Model to be sent
        """
        self.model = model
        self._pickle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Save the model with pickle
        model_pkl = BytesIO()
        torch.save(self.model, model_pkl)

        # Store it
        state['model'] = None
        state['_pickle'] = model_pkl.getvalue()
        return state

    def get_model(self, map_location: Union[str, torch.device] = 'cpu'):
        """Load the cached model into memory

        Args:
            map_location: Where to copy the device
        Returns:
            Deserialized model, moved to the target resource
        """
        if self.model is None:
            self.model = torch.load(BytesIO(self._pickle), map_location=map_location)
            self._pickle = None
        else:
            self.model.to(map_location)
        return self.model


class Moleculewise(Atomwise):

    def forward(self, inputs):
        r"""
        predicts atomwise property
        """
        atom_mask = inputs[Properties.atom_mask]

        # Pool over atoms
        inputs['representation'] = self.atom_pool(inputs['representation'], atom_mask)

        # run prediction
        y = self.out_net(inputs)
        y = self.standardize(y)

        # collect results
        result = {self.property: y}

        if self.derivative:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                result[self.property],
                inputs[Properties.R],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=self.create_graph,
                retain_graph=True,
            )[0]
            result[self.derivative] = sign * dy
        return result


class AddToRepresentation(nn.Module):
    """Add features from another source to atomic representations

    These additional features could be both for individual atoms or for the
    entire molecule. For the second case, this class will duplicate the property for each atom
    before concatenating the value with the representation.

    Properties are often taken from the input dictionary

    Args:
        additional_props ([string]): List of other properties to use as inputs
    """

    def __init__(self, additional_props: List[str]):
        super(AddToRepresentation, self).__init__()
        self.additional_props = additional_props.copy()

    def forward(self, inputs):
        # Get the representation
        rep = inputs['representation']
        n_atoms = rep.shape[1]  # Use for expanding properties

        # Append the additional props
        output = [rep]
        for p in self.additional_props:
            x = inputs[p]
            if x.dim() == 1:  # Per-molecule properties
                x = torch.unsqueeze(torch.unsqueeze(x, -1).expand(-1, n_atoms), -1)
            elif x.dim() == 2:
                x = torch.unsqueeze(x, -1)
            output.append(x)
        return torch.cat(output, -1)


class TimeoutHook(trn.Hook):
    """Hook that stop training if a timeout limit has been reached"""

    def __init__(self, timeout: float):
        """
        Args:
            timeout: Maximum allowed training time
        """
        super().__init__()
        self.timeout = timeout
        self._start_time = None

    def on_train_begin(self, trainer):
        self._start_time = monotonic()

    def on_batch_end(self, trainer, train_batch, result, loss):
        if monotonic() > self._start_time + self.timeout:
            trainer._stop = True


def evaluate_schnet(models: List[Union[TorchMessage, torch.nn.Module, Path]],
                    molecules: List[str], property_name: str,
                    batch_size: int = 64, device: str = 'cpu') -> np.ndarray:
    """Run inference for a machine learning model

    Args:
        models: List of models to evaluate. Either a SchNet model or
           the bytes corresponding to a serialized model
        molecules: XYZ-format structures of molecules to be evaluate
        property_name: Name of the property being predicted
        batch_size: Number of molecules to evaluate per batch
        device: Device on which to run the computation
    """

    # Make sure the models are converted to Torch models
    if isinstance(models[0], TorchMessage):
        models = [m.get_model(device) for m in models]
    elif isinstance(models[0], (Path, str)):
        models = [torch.load(m, map_location='cpu') for m in models]  # Load to main memory first

    # Make the dataset
    with TemporaryDirectory() as td:
        # Convert the molecules to ase.Atoms objects
        atoms = [next(read_xyz(StringIO(x), slice(None))) for x in molecules]

        # Save the data to an ASE Atoms database
        run_file = os.path.join(td, 'run_data.db')
        db = AtomsData(run_file, available_properties=[])
        db.add_systems(atoms, [{} for _ in atoms])

        # Build the data loader
        loader = AtomsLoader(db, batch_size=batch_size)

        # Run the models
        y_preds = []
        for model in models:
            y_pred = []
            model.to(device)  # Move the model to the device
            for batch in loader:
                # Push the batch to the device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Run it and save results
                pred = model(batch)
                y_pred.append(pred[property_name].detach().cpu().numpy())
            y_preds.append(np.squeeze(np.concatenate(y_pred)))

        return np.vstack(y_preds).T


def train_schnet(model: Union[TorchMessage, torch.nn.Module, Path],
                 database: Dict[str, float],
                 num_epochs: int,
                 reset_weights: bool = True,
                 property_name: str = 'output',
                 test_set: Optional[List[str]] = None,
                 device: str = 'cpu', batch_size: int = 32, validation_split: float = 0.1,
                 bootstrap: bool = False,
                 random_state: int = 1, learning_rate: float = 1e-3, patience: int = None,
                 timeout: float = None) -> Union[Tuple[TorchMessage, pd.DataFrame],
                                                 Tuple[TorchMessage, pd.DataFrame, List[float]]]:
    """Train a SchNet model

    Args:
        model: Model to be retrained
        database: Mapping of XYZ format structure to property
        num_epochs: Number of training epochs
        property_name: Name of the property being predicted
        reset_weights: Whether to re-initialize weights before training, or start training from previous
        test_set: Hold-out set. If provided, function will return the performance of the model on those weights
        device: Device (e.g., 'cuda', 'cpu') used for training
        batch_size: Batch size during training
        validation_split: Fraction to training set to use for the validation loss
        bootstrap: Whether to take a bootstrap sample of the training set before training
        random_state: Random seed used for generating validation set and bootstrap sampling
        learning_rate: Initial learning rate for optimizer
        patience: Patience until learning rate is lowered. Default: epochs / 8
        timeout: Maximum training time in seconds
    Returns:
        - model: Retrained model
        - history: Training history
        - test_pred: Predictions on ``test_set``, if provided
    """

    # Make sure the models are converted to Torch models
    if isinstance(model, TorchMessage):
        model = model.get_model(device)
    elif isinstance(model[0], (Path, str)):
        model = torch.load(model, map_location='cpu')  # Load to main memory first

    # If desired, re-initialize weights
    if reset_weights:
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    # Separate the database into molecules and properties
    xyz, y = zip(*database.items())
    xyz = np.array(xyz)
    y = np.array(y)

    # Convert the xyz files to ase Atoms
    atoms = np.array([next(read_xyz(StringIO(x), slice(None))) for x in xyz])

    # Make the training and validation splits
    rng = np.random.RandomState(random_state)
    train_split = rng.rand(len(xyz)) > validation_split
    train_X = atoms[train_split]
    train_y = y[train_split]
    valid_X = atoms[~train_split]
    valid_y = y[~train_split]

    # Perform a bootstrap sample of the training data
    if bootstrap:
        sample = rng.choice(len(train_X), size=(len(train_X),), replace=True)
        train_X = train_X[sample]
        train_y = train_y[sample]

    # Start the training process
    with TemporaryDirectory() as td:
        # Save the data to an ASE Atoms database
        train_file = os.path.join(td, 'train_data.db')
        db = AtomsData(train_file, available_properties=[property_name])
        db.add_systems(train_X, [{property_name: i} for i in train_y])
        train_loader = AtomsLoader(db, batch_size=batch_size, shuffle=True)

        valid_file = os.path.join(td, 'valid_data.db')
        db = AtomsData(valid_file, available_properties=[property_name])
        db.add_systems(valid_X, [{property_name: i} for i in valid_y])
        valid_loader = AtomsLoader(db, batch_size=batch_size)

        # Make the trainer
        opt = optim.Adam(model.parameters(), lr=learning_rate)

        loss = trn.build_mse_loss(['delta'])
        metrics = [spk.metrics.MeanSquaredError('delta')]
        if patience is None:
            patience = num_epochs // 8
        hooks = [
            trn.CSVHook(log_path=td, metrics=metrics),
            trn.ReduceLROnPlateauHook(
                opt,
                patience=patience, factor=0.8, min_lr=1e-6,
                stop_after_min=True
            )
        ]

        if timeout is not None:
            hooks.append(TimeoutHook(timeout))

        trainer = trn.Trainer(
            model_path=td,
            model=model,
            hooks=hooks,
            loss_fn=loss,
            optimizer=opt,
            train_loader=train_loader,
            validation_loader=valid_loader,
            checkpoint_interval=num_epochs + 1  # Turns off checkpointing
        )

        trainer.train(device, n_epochs=num_epochs)

        # Load in the best model
        model = torch.load(os.path.join(td, 'best_model'))

        # If desired, report the performance on a test set
        test_pred = None
        if test_set is not None:
            test_pred = evaluate_schnet([model], test_set,
                                        property_name=property_name, batch_size=batch_size, device=device)

        # Move the model off of the GPU to save memory
        if 'cuda' in device:
            model.to('cpu')

        # Load in the training results
        train_results = pd.read_csv(os.path.join(td, 'log.csv'))

        # Return the results
        if test_pred is None:
            return TorchMessage(model), train_results
        else:
            return TorchMessage(model), train_results, test_pred[:, 0].tolist()
