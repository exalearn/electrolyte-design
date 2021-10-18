"""Test data"""
import tensorflow as tf

from moldesign.score.mpnn.data import make_tfrecord, make_data_loader
from moldesign.utils.conversions import convert_string_to_nx, convert_nx_to_dict


def test_preprocess_and_loader(tmpdir, dataset):
    smiles, multis, scalars = dataset

    # Convert to needed formats
    nxs = [convert_string_to_nx(s) for s in smiles]

    # Save data to a temporary directory
    data_path = tmpdir.join('temp.proto')
    with tf.io.TFRecordWriter(str(data_path)) as writer:
        for n, m, s in zip(nxs, multis, scalars):
            record = convert_nx_to_dict(n)
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
