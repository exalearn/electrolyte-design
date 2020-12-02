"""Test data"""
from typing import Tuple, List

import tensorflow as tf
from pytest import fixture

from moldesign.score.mpnn.data import make_type_lookup_tables, make_tfrecord, make_data_loader
from moldesign.utils.conversions import convert_smiles_to_nx, convert_nx_to_dict


@fixture()
def dataset() -> Tuple[List[str], List[List[float]], List[float]]:
    return ['C', 'CC'], [[1., 2.], [2., 3.]], [1., 2.]


def test_preprocess_and_loader(tmpdir, dataset):
    smiles, multis, scalars = dataset

    # Convert to needed formats
    nxs = [convert_smiles_to_nx(s, add_hs=True) for s in smiles]

    # Save data as the
    atom_types, bond_types = make_type_lookup_tables(nxs)
    assert atom_types == [1, 6]
    assert bond_types == ['SINGLE']

    # Save data to a temporary directory
    data_path = tmpdir.join('temp.proto')
    with tf.io.TFRecordWriter(str(data_path)) as writer:
        for n, m, s in zip(nxs, multis, scalars):
            record = convert_nx_to_dict(n, atom_types, bond_types)
            record['multi'] = m
            record['scalar'] = s
            writer.write(make_tfrecord(record))
    assert data_path.isfile()

    # Make a data loader with the multi-property output
    loader = make_data_loader(str(data_path), batch_size=1, output_property='multi', output_shape=(2,))
    ins, outs = next(iter(loader))
    assert ins['atom'].shape == (5,)
    assert outs.shape == (1, 2)

    # Make the data loader with the scalar output
    loader = make_data_loader(str(data_path), batch_size=1, output_property='scalar')
    ins, outs = next(iter(loader))
    assert ins['atom'].shape == (5,)
    assert outs.shape == (1,)
