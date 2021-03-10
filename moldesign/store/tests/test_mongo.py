"""Test for the MongoDB wrapper"""
from pathlib import Path

from pymongo import MongoClient
from pytest import fixture
from qcelemental.models import OptimizationResult

from moldesign.store.models import MoleculeData
from moldesign.store.mongo import generate_update, MoleculePropertyDB

_my_path = Path(__file__).parent


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
    md = MoleculeData.from_identifier('O')
    result = OptimizationResult.parse_file(_my_path.joinpath('records/xtb-neutral.json'))
    md.add_geometry(result, "xtb")
    md.subsets.append('pytest')
    return md


@fixture
def sample_db(init_db, sample_record) -> MoleculePropertyDB:
    init_db.update_molecule(sample_record)
    return init_db


def test_generate_update(sample_record):
    update = generate_update(sample_record)
    assert len(update) == 2
    assert 'data.xtb.neutral.atomization_energy.xtb-no_zpe' in update['$set']
    assert update['$set']['identifier.smiles'] == "O"
    assert update['$addToSet']['subsets'] == {"$each": ['pytest']}


def test_initialize(db):
    db.initialize_index()
    assert db.collection.index_information()['key_1']['unique']


def test_training_set(init_db, sample_record):
    init_db.update_molecule(sample_record)
    output = init_db.get_training_set(['identifier.smiles'], ['data.xtb.neutral.atomization_energy.xtb-no_zpe'])
    assert output['identifier.smiles'] == ['O']
    assert output['data.xtb.neutral.atomization_energy.xtb-no_zpe'][0] > -1


def test_retrieve_molecules(sample_db):
    assert sample_db.get_molecules() == {'InChI=1S/H2O/h1H2'}


def test_eligible_molecules(sample_db):
    records = sample_db.get_eligible_molecules(['identifier.smiles'], ['data.xtb.neutral.atomization_energy.g4mp2'])
    assert len(records['key']) == 1
    assert records['identifier.smiles'] == ['O']

    records = sample_db.get_eligible_molecules(['identifier.inchi', 'data.xtb.neutral.atomization_energy.xtb-no_zpe'],
                                               ['data.xtb.neutral.atomization_energy.g4mp2'])
    assert records['data.xtb.neutral.atomization_energy.xtb-no_zpe'] == [-0.5159719695680218]


def test_get_record(sample_db):
    record = sample_db.get_molecule_record(smiles='O', projection=["identifier.smiles", "key", "subsets"])
    assert record is not None
    assert record.subsets == ["pytest"]
    assert record.identifier["smiles"] == "O"

    record = sample_db.get_molecule_record(smiles='CC')
    assert record is not None
