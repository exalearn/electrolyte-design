"""Functions for updating and performing bulk inference using an Keras MPNN model"""
from pathlib import Path
from functools import partial
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks as cb

from moldesign.utils.globals import get_process_pool
from moldesign.score.mpnn.callbacks import LRLogger, EpochTimeLogger, TimeLimitCallback
from moldesign.score.mpnn.data import _merge_batch, GraphLoader
from moldesign.score.mpnn.layers import custom_objects
from moldesign.utils.conversions import convert_string_to_dict

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
                  smiles: Union[List[str], List[dict]],
                  batch_size: int = 128, cache: bool = True, n_jobs: Optional[int] = 1) -> np.ndarray:
    """Run inference on a list of molecules

    Args:
        model_msg: List of MPNNs to evaluate. Accepts either a pickled message, model, or a path
        smiles: List of molecules to evaluate either as SMILES or InChI strings, or lists of MPNN-ready dictionary objections
        batch_size: Number of molecules per batch
        cache: Whether to cache models if being read from disk
        n_jobs: Number of parallel jobs to run. Set `None` to use total number of cores
            Note: The Pool is cached, so the first value of n_jobs is set to will remain
            for the life of the process (except if the value is 1, which does not use a Pool)
    Returns:
        Predicted value for each molecule
    """
    assert len(smiles) > 0, "You must provide at least one molecule to inference function"

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

    # Ensure all molecules are ready for inference
    if isinstance(smiles[0], dict):
        mols = smiles
    else:
        if n_jobs == 1:
            mols = [convert_string_to_dict(s) for s in smiles]
        else:
            pool = get_process_pool(n_jobs)
            mols = pool.map(convert_string_to_dict, smiles)

    # Stuff them into batches
    chunks = [mols[start:start + batch_size] for start in range(0, len(mols), batch_size)]
    batches = [_merge_batch(c) for c in chunks]

    # Feed the batches through the MPNN
    all_outputs = []
    for model in models:
        outputs = [model(b) for b in batches]
        all_outputs.append(np.vstack(outputs))
    return np.hstack(all_outputs)


def update_mpnn(model: Union[MPNNMessage, tf.keras.Model, Path, str],
                database: Dict[str, float], num_epochs: int, test_set: Optional[List[str]] = None,
                batch_size: int = 32, validation_split: float = 0.1, bootstrap: bool = False,
                random_state: int = 1, learning_rate: float = 1e-3, patience: int = None,
                timeout: float = None)\
        -> Union[Tuple[List, dict], Tuple[List, dict, List[float]]]:
    """Update a model with new training sets

    Args:
        model: Serialized version of the model, the model to be retrain, or path to it on disk
        database: Training dataset of molecule mapped to a property
        num_epochs: Maximum number of epochs to run
        test_set: Hold-out set. If provided, this function will return predictions on this set
        batch_size: Number of molecules per training batch
        validation_split: Fraction of molecules used for the training/validation split
        bootstrap: Whether to perform a bootstrap sample of the dataset
        random_state: Seed to the random number generator. Ensures entries do not move between train
            and validation set as the database becomes larger
        learning_rate: Learning rate for the Adam optimizer
        patience: Number of epochs without improvement before terminating training.
        timeout: Maximum training time in seconds
    Returns:
        model: Updated weights
        history: Training history
    """

    if isinstance(model, MPNNMessage):
        # Rebuild the model from message
        model = model.get_model()
    elif isinstance(model, (Path, str)):
        model = tf.keras.models.load_model(model, custom_objects=custom_objects)

    return _train_model(model, database, num_epochs, test_set, batch_size, validation_split,
                        bootstrap, random_state, learning_rate, patience, timeout)


def retrain_mpnn(model_config: dict, database: Dict[str, float], num_epochs: int, test_set: Optional[List[str]] = None,
                 batch_size: int = 32, validation_split: float = 0.1, bootstrap: bool = False,
                 random_state: int = 1, learning_rate: float = 1e-3,
                 patience: int = None, timeout: float = None)\
        -> Union[Tuple[List, dict], Tuple[List, dict, List[float]]]:
    """Train a model from initialized weights

    Args:
        model_config: Serialized version of the model
        database: Training dataset of molecule mapped to a property
        num_epochs: Maximum number of epochs to run
        test_set: Hold-out set. If provided, this function will return predictions on this set
        batch_size: Number of molecules per training batch
        validation_split: Fraction of molecules used for the training/validation split
        bootstrap: Whether to perform a bootstrap sample of the dataset
        random_state: Seed to the random number generator. Ensures entries do not move between train
            and validation set as the database becomes larger
        learning_rate: Learning rate for the Adam optimizer
        patience: Number of epochs without improvement before terminating training.
        timeout: Maximum training time in seconds
    Returns:
        - model: Updated weights
        - history: Training history
        - test_pred: Prediction on test set, if provided
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

    return _train_model(model, database, num_epochs, test_set, batch_size, validation_split,
                        bootstrap, random_state, learning_rate, patience, timeout)


def _train_model(model: tf.keras.Model, database: Dict[str, float], num_epochs: int, test_set: Optional[List[str]],
                 batch_size: int = 32, validation_split: float = 0.1, bootstrap: bool = False,
                 random_state: int = 1, learning_rate: float = 1e-3, patience: int = None,
                 timeout: float = None) -> Union[Tuple[List, dict], Tuple[List, dict, List[float]]]:
    """Train a model

    Args:
        model: Model to be trained
        database: Training dataset of molecule mapped to a property
        test_set: Hold-out set. If provided, this function will return predictions on this set
        num_epochs: Maximum number of epochs to run
        batch_size: Number of molecules per training batch
        validation_split: Fraction of molecules used for the training/validation split
        bootstrap: Whether to perform a bootstrap sample of the dataset
        random_state: Seed to the random number generator. Ensures entries do not move between train
            and validation set as the database becomes larger
        learning_rate: Learning rate for the Adam optimizer
        patience: Number of epochs without improvement before terminating training.
        timeout: Maximum training time in seconds
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
    train_loader = GraphLoader(train_X, train_y, batch_size=batch_size, shuffle=True)
    val_loader = GraphLoader(valid_X, valid_y, batch_size=batch_size, shuffle=False)

    # Make the callbacks
    final_learn_rate = 1e-6
    init_learn_rate = learning_rate
    decay_rate = (final_learn_rate / init_learn_rate) ** (1. / (num_epochs - 1))

    def lr_schedule(epoch, lr):
        return lr * decay_rate

    if patience is None:
        patience = num_epochs // 8

    early_stopping = cb.EarlyStopping(patience=patience, restore_best_weights=True)
    my_callbacks = [
        LRLogger(),
        EpochTimeLogger(),
        cb.LearningRateScheduler(lr_schedule),
        early_stopping,
        cb.TerminateOnNaN(),
        train_loader  # So the shuffling gets called
    ]
    if timeout is not None:
        my_callbacks += [
            TimeLimitCallback(timeout)
        ]

    # Run the desired number of epochs
    history = model.fit(train_loader, epochs=num_epochs, validation_data=val_loader,
                        verbose=False, shuffle=False, callbacks=my_callbacks)

    # If a timeout is used, make sure we are using the best weights
    #  The training may have exited without storing the best weights
    if timeout is not None:
        model.set_weights(early_stopping.best_weights)

    # Check if there is a NaN loss
    if np.isnan(history.history['loss']).any():
        raise ValueError('Training failed due to a NaN loss.')

    # If provided, evaluate model on test set
    test_pred = None
    if test_set is not None:
        test_pred = evaluate_mpnn([model], test_set, batch_size, cache=False)

    # Convert weights to numpy arrays (avoids mmap issues)
    weights = []
    for v in model.get_weights():
        v = np.array(v)
        if np.isnan(v).any():
            raise ValueError('Found some NaN weights.')
        weights.append(v)

    # Once we are finished training call "clear_session"
    tf.keras.backend.clear_session()
    if test_pred is None:
        return weights, history.history
    else:
        return weights, history.history, test_pred[:, 0].tolist()
