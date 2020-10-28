"""Functions for updating and performing bulk inference using an Keras MPNN model"""
from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf

from moldesign.score.mpnn.data import convert_nx_to_dict, _merge_batch, GraphLoader
from moldesign.score.mpnn.layers import custom_objects
from moldesign.utils.conversions import convert_smiles_to_nx


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

    def get_model(self) -> tf.keras.Model:
        model = tf.keras.models.model_from_json(self.config, custom_objects=custom_objects)
        model.set_weights(self.weights)
        return model


def evaluate_mpnn(model_msg: List[MPNNMessage], smiles: List[str],
                  atom_types: List[int], bond_types: List[str], batch_size: int = 128) -> np.ndarray:
    """Run inference on a list of molecules

    Args:
        model_msg: List of MPNNs to evaluate
        smiles: List of molecules to evaluate
        atom_types: List of known atom types
        bond_types: List of known bond types
        batch_size: List of molecules to create into matches
    Returns:
        Predicted value for each molecule
    """

    # Rebuild the model
    tf.keras.backend.clear_session()
    models = [m.get_model() for m in model_msg]

    # Convert all SMILES strings to batches of molecules
    # TODO (wardlt): Use multiprocessing. Could benefit from a persistent Pool to avoid loading in TF many times
    mols = [convert_nx_to_dict(convert_smiles_to_nx(s), atom_types, bond_types) for s in smiles]
    chunks = [mols[start:start + batch_size] for start in range(0, len(mols), batch_size)]
    batches = [_merge_batch(c) for c in chunks]

    # Feed the batches through the MPNN
    all_outputs = []
    for model in models:
        outputs = [model.predict_on_batch(b) for b in batches]
        all_outputs.append(np.vstack(outputs))
    return np.hstack(all_outputs)


# TODO (wardlt): Evaluate whether the model stays in memory after training. If so, clear graph?
def update_mpnn(model_msg: MPNNMessage, database: Dict[str, float], num_epochs: int,
                atom_types: List[int], bond_types: List[str], batch_size: int = 512,
                validation_split: float = 0.1, random_state: int = 1, learning_rate: float = 1e-3)\
        -> Tuple[List, dict]:
    """Update a model with new training sets

    Args:
        model_msg: Serialized version of the model
        database: Training dataset of molecule mapped to a property
        atom_types: List of known atom types
        bond_types: List of known bond types
        num_epochs: Number of epochs to run
        batch_size: Number of molecules per training batch
        validation_split: Fraction of molecules used for the training/validation split
        random_state: Seed to the random number generator. Ensures entries do not move between train
            and validation set as the database becomes larger
        learning_rate: Learning rate for the Adam optimizer
    Returns:
        model: Updated weights
        history: Training history
    """

    # Rebuild the model
    tf.keras.backend.clear_session()
    model = model_msg.get_model()
    model.compile(tf.keras.optimizers.Adam(lr=learning_rate), 'mean_absolute_error')

    # Separate the database into molecules and properties
    smiles, y = zip(*database.items())

    # Make the training and validation splits
    #  Use a random number generator with fixed seed to ensure that the validation
    #  set is never polluted with entries from the training set
    # TODO (wardlt): Replace with passing train and validation separately?
    rng = np.random.RandomState(random_state)
    train_split = rng.rand(len(smiles)) > validation_split

    # Make the loaders
    smiles = np.array(smiles)
    y = np.array(y)
    train_loader = GraphLoader(smiles[train_split], atom_types, bond_types, y[train_split],
                               batch_size=batch_size)
    val_loader = GraphLoader(smiles[~train_split], atom_types, bond_types, y[~train_split],
                             batch_size=batch_size, shuffle=False)

    # Run the desired number of epochs
    # TODO (wardlt): Should we use callbacks to get only the "best model" based on the validation set?
    history = model.fit(train_loader, epochs=num_epochs, validation_data=val_loader, verbose=False)
    return [np.array(v) for v in model.get_weights()], history.history