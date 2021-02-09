from typing import Tuple, List, Dict

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from pytest import fixture

from moldesign.score.mpnn.layers import Squeeze, GraphNetwork


@fixture()
def dataset() -> Tuple[List[str], List[List[float]], List[float]]:
    return ['C', 'CC'], [[1., 2.], [2., 3.]], [1., 2.]


@fixture()
def train_dataset() -> Dict[str, float]:
    return {'C': 1., 'CC': 2., 'CCC': 3., 'CCCC': 4.0}


@fixture()
def atom_types() -> List[int]:
    return [1, 6]


@fixture()
def bond_types() -> List[str]:
    return ['SINGLE']


@fixture()
def model() -> Model:
    node_graph_indices = Input(shape=(1,), name='node_graph_indices', dtype='int32')
    atom_types = Input(shape=(1,), name='atom', dtype='int32')
    bond_types = Input(shape=(1,), name='bond', dtype='int32')
    connectivity = Input(shape=(2,), name='connectivity', dtype='int32')

    # Squeeze the node graph and connectivity matrices
    snode_graph_indices = Squeeze(axis=1)(node_graph_indices)
    satom_types = Squeeze(axis=1)(atom_types)
    sbond_types = Squeeze(axis=1)(bond_types)

    output = GraphNetwork(2, 1, 16, 1, atomic_contribution=True, reduce_function='sum',
                          name='mpnn')([satom_types, sbond_types, snode_graph_indices, connectivity])

    # Scale the output
    output = Dense(1, activation='linear', name='scale')(output)

    return Model(inputs=[node_graph_indices, atom_types, bond_types, connectivity],
                 outputs=output)
