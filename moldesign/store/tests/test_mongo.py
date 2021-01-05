"""Test for the MongoDB wrapper"""
from pymongo import MongoClient
from pytest import fixture

from moldesign.store.models import MoleculeData
from moldesign.store.mongo import generate_update, MoleculePropertyDB


@fixture
def db() -> MoleculePropertyDB:
    client = MongoClient()
    db = client['edw-pytest']
    yield MoleculePropertyDB(db['molecules'])
    client.drop_database('edw-pytest')


@fixture
def init_db(db) -> MoleculePropertyDB:
    """Make a pre-initialized database"""
    db.initialize_index()
    return db


@fixture
def sample_record() -> MoleculeData:
    md = MoleculeData.from_identifier('C')
    md.atomization_energy['small_basis'] = -1
    return md


def test_generate_update(sample_record):
    assert generate_update(sample_record) == {
        '$set': {'atomization_energy.small_basis': -1, 'identifiers.smiles': 'C', 'subsets': []}
    }


def test_initialize(db):
    db.initialize_index()
    assert db.collection.index_information()['key_1'] == {'v': 2, 'unique': True, 'key': [('key', 1)],
                                                          'ns': 'edw-pytest.molecules'}


def test_training_set(init_db, sample_record):
    init_db.update_molecule(sample_record)
    inputs, outputs = init_db.get_training_set(['identifiers.smiles'],
                                               ['atomization_energy.small_basis'])
    assert inputs == {'identifiers.smiles': ['C']}
