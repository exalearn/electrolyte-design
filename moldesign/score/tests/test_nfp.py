import numpy as np

from moldesign.score.nfp import make_data_loader


def test_loader():
    # Test a basic loader
    smiles = ['C', 'CC', 'CCC']
    loader = make_data_loader(smiles, batch_size=2)
    batch = next(loader.take(1).as_numpy_iterator())

    assert np.equal(batch['atom'], np.array([[6, 1, 1, 1, 1, 0, 0, 0],  # CH4
                                             [6, 6, 1, 1, 1, 1, 1, 1]])).all()  # C2H6

    # Test one where we give some input values
    loader = make_data_loader(smiles, values=[1., 2., 3.], batch_size=2)
    batch = next(loader.take(1).as_numpy_iterator())
    assert np.equal(batch[0]['atom'], np.array([[6, 1, 1, 1, 1, 0, 0, 0],  # CH4
                                                [6, 6, 1, 1, 1, 1, 1, 1]])).all()  # C2H6
    assert np.equal(batch[1], np.array([1., 2.])).all()

    # Test where I fix the number of atoms
    loader = make_data_loader(smiles, values=[1., 2., 3.], batch_size=2, max_size=20)
    batch = next(loader.take(1).as_numpy_iterator())
    assert batch[0]['bond'].shape == (2, 80)

    # Test with shuffling
    loader = make_data_loader(smiles, batch_size=2, max_size=20, shuffle_buffer=4)
    batch = next(loader.take(1).as_numpy_iterator())
    assert batch['bond'].shape == (2, 80)

    # Test with dropping the last batch
    #  Should result in a spec with a fixed size in every dimension
    loader = make_data_loader(smiles, batch_size=2, max_size=20, shuffle_buffer=4, drop_last_batch=True)
    for key, spec in loader.element_spec.items():
        assert all(x is not None for x in spec.shape), key
