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
    db.drop_collection('molecules')
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
    md.subsets.append('pytest')
    return md


@fixture
def sample_db(init_db, sample_record) -> MoleculePropertyDB:
    init_db.update_molecule(sample_record)
    return init_db


def test_generate_update(sample_record):
    update = generate_update(sample_record)
    assert len(update) == 2
    assert update['$set']['atomization_energy.small_basis'] == -1
    assert update['$set']['identifier.smiles'] == "C"
    assert update['$addToSet']['subsets'] == {"$each": ['pytest']}


def test_initialize(db):
    db.initialize_index()
    assert db.collection.index_information()['key_1']['unique']


def test_training_set(init_db, sample_record):
    init_db.update_molecule(sample_record)
    output = init_db.get_training_set(['identifier.smiles'], ['atomization_energy.small_basis'])
    assert output['identifier.smiles'] == ['C']
    assert output['atomization_energy.small_basis'] == [-1]


def test_retrieve_molecules(sample_db):
    assert sample_db.get_molecules() == {'InChI=1S/CH4/h1H4'}


def test_eligible_molecules(sample_db):
    records = sample_db.get_eligible_molecules(['identifier.smiles'], ['atomization_energy.g4mp2'])
    assert len(records['key']) == 1
    assert records['identifier.smiles'] == ['C']

    records = sample_db.get_eligible_molecules(['identifier.inchi', 'atomization_energy.small_basis'],
                                               ['atomization_energy.g4mp2'])
    assert records['atomization_energy.small_basis'] == [-1]


def test_get_record(sample_db):
    record = sample_db.get_molecule_record(smiles='C', projection=["identifier.smiles", "key", "subsets"])
    assert record is not None
    assert record.subsets == ["pytest"]
    assert record.identifier["smiles"] == "C"

    record = sample_db.get_molecule_record(smiles='CC')
    assert record is not None
