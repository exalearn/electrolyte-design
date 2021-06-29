"""Functions for updating and performing bulk inference using an Keras MPNN model"""
from pathlib import Path
from functools import partial
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks as cb

from moldesign.utils.globals import get_process_pool
from moldesign.score.mpnn.callbacks import LRLogger, EpochTimeLogger
from moldesign.score.mpnn.data import _merge_batch, GraphLoader
from moldesign.score.mpnn.layers import custom_objects
from moldesign.utils.conversions import convert_smiles_to_dict


# Process-local caches
_model_cache = {}  # Models loaded from disk


# TODO (wardlt): Make this Keras message object usable elsewhere
class MPNNMessage:
    """Package for sending an MPNN model over connections that require pickling"""

    def __init__(self, model: tf.keras.Model):
        """
        Args:
            model: Model to be sent
        """

        self.config = model.to_json()
        # Makes a copy of the weights to ensure they are not memoryview objects
        self.weights = [np.array(v) for v in model.get_weights()]

        # Cached copy of the model
        self._model = None

    def __getstate__(self):
        """Get all of state except the model"""
        state = self.__dict__.copy()
        state['_model'] = None
        return state

    def get_model(self) -> tf.keras.Model:
        """Get a copy of the model

        Returns:
            The model specified by this message
        """
        if self._model is None:
            self._model = tf.keras.models.model_from_json(self.config, custom_objects=custom_objects)
            self._model.set_weights(self.weights)
        return self._model


# TODO (wardlt): Split into multiple functions? I don't like having so many input type options
def evaluate_mpnn(model_msg: Union[List[MPNNMessage], List[tf.keras.Model], List[str], List[Path]],
                  smiles: List[str], atom_types: List[int], bond_types: List[str],
                  batch_size: int = 128, cache: bool = True, n_jobs: Optional[int] = 1) -> np.ndarray:
    """Run inference on a list of molecules

    Args:
        model_msg: List of MPNNs to evaluate. Accepts either a pickled message, model, or a path
        smiles: List of molecules to evaluate
        atom_types: List of known atom types
        bond_types: List of known bond types
        batch_size: Number of molecules per batch
        cache: Whether to cache models if being read from disk
        n_jobs: Number of parallel jobs to run. Set `None` to use total number of cores
            Note: The Pool is cached, so the first value of n_jobs is set to will remain
            for the life of the process (except if the value is 1, which does not use a Pool)
    Returns:
        Predicted value for each molecule
    """

    # Access the model
    if isinstance(model_msg[0], MPNNMessage):
        # Unpack the messages
        models = [m.get_model() for m in model_msg]
    elif isinstance(model_msg[0], (str, Path)):
        # Load the model from disk
        if cache:
            models = []
            for p in model_msg:
                if p not in _model_cache:
                    _model_cache[p] = tf.keras.models.load_model(str(p), custom_objects=custom_objects)
                models.append(_model_cache[p])
        else:
            models = [tf.keras.models.load_model(p, custom_objects=custom_objects)
                      for p in model_msg]
    else:
        # No action needed
        models = model_msg

    # Convert all SMILES strings to batches of molecules
    if n_jobs == 1:
        mols = [convert_smiles_to_dict(s, atom_types, bond_types, add_hs=True) for s in smiles]
    else:
        pool = get_process_pool(n_jobs)
        f = partial(convert_smiles_to_dict, atom_types=atom_types, bond_types=bond_types, add_hs=True)
        mols = pool.map(f, smiles)
    chunks = [mols[start:start + batch_size] for start in range(0, len(mols), batch_size)]
    batches = [_merge_batch(c) for c in chunks]

    # Feed the batches through the MPNN
    all_outputs = []
    for model in models:
        outputs = [model.predict_on_batch(b) for b in batches]
        all_outputs.append(np.vstack(outputs))
    return np.hstack(all_outputs)


def update_mpnn(model_msg: MPNNMessage, database: Dict[str, float], num_epochs: int,
                atom_types: List[int], bond_types: List[str], batch_size: int = 32,
                validation_split: float = 0.1, bootstrap: bool = False,
                random_state: int = 1, learning_rate: float = 1e-3, patience: int = None)\
        -> Tuple[List, dict]:
    """Update a model with new training sets

    Args:
        model_msg: Serialized version of the model
        database: Training dataset of molecule mapped to a property
        atom_types: List of known atom types
        bond_types: List of known bond types
        num_epochs: Maximum number of epochs to run
        batch_size: Number of molecules per training batch
        validation_split: Fraction of molecules used for the training/validation split
        bootstrap: Whether to perform a bootstrap sample of the dataset
        random_state: Seed to the random number generator. Ensures entries do not move between train
            and validation set as the database becomes larger
        learning_rate: Learning rate for the Adam optimizer
        patience: Number of epochs without improvement before terminating training.
    Returns:
        model: Updated weights
        history: Training history
    """

    # Rebuild the model from message
    model = model_msg.get_model()
    return _train_model(model, database, num_epochs, atom_types, bond_types, batch_size, validation_split,
                        bootstrap, random_state, learning_rate, patience)


def retrain_mpnn(model_config: dict, database: Dict[str, float], num_epochs: int,
                 atom_types: List[int], bond_types: List[str], batch_size: int = 32,
                 validation_split: float = 0.1, bootstrap: bool = False,
                 random_state: int = 1, learning_rate: float = 1e-3,
                 patience: int = None) -> Tuple[List, dict]:
    """Train a model from initialized weights

    Args:
        model_config: Serialized version of the model
        database: Training dataset of molecule mapped to a property
        atom_types: List of known atom types
        bond_types: List of known bond types
        num_epochs: Maximum number of epochs to run
        batch_size: Number of molecules per training batch
        validation_split: Fraction of molecules used for the training/validation split
        bootstrap: Whether to perform a bootstrap sample of the dataset
        random_state: Seed to the random number generator. Ensures entries do not move between train
            and validation set as the database becomes larger
        learning_rate: Learning rate for the Adam optimizer
        patience: Number of epochs without improvement before terminating training.
    Returns:
        model: Updated weights
        history: Training history
    """

    # Make a copy of the model
    model = tf.keras.models.Model.from_config(model_config, custom_objects=custom_objects)

    # Define initial guesses for the "scaling" later
    try:
        scale_layer = model.get_layer('scale')
        outputs = np.array(list(database.values()))
        scale_layer.set_weights([outputs.std()[None, None], outputs.mean()[None]])
    except ValueError:
        pass

    return _train_model(model, database, num_epochs, atom_types, bond_types, batch_size, validation_split,
                        bootstrap, random_state, learning_rate, patience)


def _train_model(model: tf.keras.Model, database: Dict[str, float], num_epochs: int,
                 atom_types: List[int], bond_types: List[str], batch_size: int = 32,
                 validation_split: float = 0.1, bootstrap: bool = False,
                 random_state: int = 1, learning_rate: float = 1e-3,
                 patience: int = None) -> Tuple[List, dict]:
    """Train a model

    Args:
        model: Model to be trained
        database: Training dataset of molecule mapped to a property
        atom_types: List of known atom types
        bond_types: List of known bond types
        num_epochs: Maximum number of epochs to run
        batch_size: Number of molecules per training batch
        validation_split: Fraction of molecules used for the training/validation split
        bootstrap: Whether to perform a bootstrap sample of the dataset
        random_state: Seed to the random number generator. Ensures entries do not move between train
            and validation set as the database becomes larger
        learning_rate: Learning rate for the Adam optimizer
        patience: Number of epochs without improvement before terminating training.
    Returns:
        model: Updated weights
        history: Training history
    """
    # Compile the model with a new optimizer
    #  We find that it is best to reset the optimizer before updating
    model.compile(tf.keras.optimizers.Adam(lr=learning_rate), 'mean_squared_error')

    # Separate the database into molecules and properties
    smiles, y = zip(*database.items())
    smiles = np.array(smiles)
    y = np.array(y)

    # Make the training and validation splits
    rng = np.random.RandomState(random_state)
    train_split = rng.rand(len(smiles)) > validation_split
    train_X = smiles[train_split]
    train_y = y[train_split]
    valid_X = smiles[~train_split]
    valid_y = y[~train_split]

    # Perform a bootstrap sample of the training data
    if bootstrap:
        sample = rng.choice(len(train_X), size=(len(train_X),), replace=True)
        train_X = train_X[sample]
        train_y = train_y[sample]

    # Make the training data loaders
    train_loader = GraphLoader(train_X, atom_types, bond_types, train_y, batch_size=batch_size, shuffle=True)
    val_loader = GraphLoader(valid_X, atom_types, bond_types, valid_y, batch_size=batch_size, shuffle=False)

    # Make the callbacks
    final_learn_rate = 1e-6
    init_learn_rate = learning_rate
    decay_rate = (final_learn_rate / init_learn_rate) ** (1. / (num_epochs - 1))

    def lr_schedule(epoch, lr):
        return lr * decay_rate

    if patience is None:
        patience = num_epochs // 8

    my_callbacks = [
        LRLogger(),
        EpochTimeLogger(),
        cb.LearningRateScheduler(lr_schedule),
        cb.EarlyStopping(patience=patience, restore_best_weights=True),
        cb.TerminateOnNaN(),
        train_loader  # So the shuffling gets called
    ]

    # Run the desired number of epochs
    history = model.fit(train_loader, epochs=num_epochs, validation_data=val_loader,
                        verbose=False, shuffle=False, callbacks=my_callbacks)

    # Check if there is a NaN loss
    if np.isnan(history.history['loss']).any():
        raise ValueError('Training failed due to a NaN loss.')

    # Convert weights to numpy arrays (avoids mmap issues)
    weights = []
    for v in model.get_weights():
        v = np.array(v)
        if np.isnan(v).any():
            raise ValueError('Found some NaN weights.')
        weights.append(v)

    # Once we are finished training call "clear_session"
    tf.keras.backend.clear_session()
    return weights, history.history
