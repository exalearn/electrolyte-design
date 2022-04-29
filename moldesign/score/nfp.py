"""Interfaces to run `NFP <https://github.com/NREL/nfp>`_ models through Colmena workflows"""
from typing import List, Any, Optional, Tuple

import tensorflow as tf
import numpy as np

from moldesign.utils.conversions import convert_string_to_dict


def _to_nfp_dict(x: dict) -> dict:
    """Convert a moldesign-compatible dict to one usable by NFP

    Removes the ``n_atom`` and ``n_bond`` keys, and increments the bond type and atom type by
    1 because nfp uses 0 as a padding mask

    Args:
        x: Dictionary to be modified
    Returns:
        The input dictionary
    """

    for k in ['atom', 'bond']:
        x[k] = np.add(x[k], 1)
    del x['n_atom']
    del x['n_bond']
    return x


def make_data_loader(mols: List[str],
                     values: Optional[List[Any]] = None,
                     batch_size: int = 32,
                     shuffle_buffer: Optional[int] = None,
                     value_spec: tf.TensorSpec = tf.TensorSpec((), dtype=tf.float32),
                     max_size: Optional[int] = None,
                     drop_last_batch: bool = False) -> tf.data.Dataset:
    """Make a data loader for data compatible with NFP-style neural networks

    Args:
        mols: List of molecules in a string format
        values: List of output values, if included in the output
        value_spec: Tensorflow specification for the output
        batch_size: Number of molecules per batch
        shuffle_buffer: Size of a shuffle buffer. Use ``None`` to leave data unshuffled
        max_size: Maximum number of atoms per molecule
        drop_last_batch: Whether to keep the last batch in the dataset. Set to ``True`` if, for example, you need every batch to be the same size
    Returns:
        Data loader that generates molecules in the desired shapes
    """

    # Convert the molecules to dictionary formats
    mol_dicts = [_to_nfp_dict(convert_string_to_dict(s)) for s in mols]

    # Make the initial data loader
    record_sig = {
        "atom": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "bond": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "connectivity": tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
    }
    if values is None:
        def generator():
            yield from mol_dicts
    else:
        def generator():
            yield from zip(mol_dicts, values)

        record_sig = (record_sig, value_spec)

    loader = tf.data.Dataset.from_generator(generator=generator, output_signature=record_sig).cache()  # TODO (wardlt): Make caching optional?

    # Shuffle, if desired
    if shuffle_buffer is not None:
        loader = loader.shuffle(shuffle_buffer)

    # Make the batches
    if max_size is None:
        loader = loader.padded_batch(batch_size=batch_size, drop_remainder=drop_last_batch)
    else:
        max_bonds = 4 * max_size  # If all atoms are carbons, they each have 4 points
        padded_records = {
            "atom": tf.TensorShape((max_size,)),
            "bond": tf.TensorShape((max_bonds,)),
            "connectivity": tf.TensorShape((max_bonds, 2))
        }
        if values is not None:
            padded_records = (padded_records, value_spec.shape)
        loader = loader.padded_batch(batch_size=batch_size, padded_shapes=padded_records, drop_remainder=drop_last_batch)

    return loader

