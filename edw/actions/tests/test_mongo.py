from pymongo import MongoClient
from edw.actions import mongo
from pytest import fixture
from gridfs import GridFS
import os

files_dir = os.path.dirname(__file__)


@fixture
def client():
    return MongoClient('localhost')


@fixture
def collection(client):
    client.drop_database('jcesr_test')
    return mongo.initialize_collection(client, 'jcesr_test')


@fixture
def gridfs(client, collection):
    return GridFS(client.get_database('jcesr_test'))


def test_add(collection):
    assert mongo.add_molecule(collection, 'C')
    assert not mongo.add_molecule(collection, 'C')
    key = mongo.compute_inchi_key('C')
    record = collection.find_one(filter={'inchi_key': key})
    assert record['inchi_key'] == key
    assert record['identifiers'] == {'smiles': 'C'}


def test_add_geometry(collection):
    inchi_key = mongo.compute_inchi_key('C')
    mongo.add_molecule(collection, 'C')
    mongo.add_geometry(collection, inchi_key, 'test', 'Hello XYZ')
    record = collection.find_one({'inchi_key': inchi_key})
    assert record['geometries'] == {'test': 'Hello XYZ'}


def test_add_calculation(collection, gridfs):
    inchi_key = mongo.compute_inchi_key('C')
    mongo.add_molecule(collection, 'C')
    mongo.add_calculation(collection, gridfs, inchi_key,
                          'test', 'Junk', 'More Junk', 'Gaussian')
    record = collection.find_one({'inchi_key': inchi_key})
    assert 'test' in record['calculations']
    assert gridfs.get(record['calculations']['test']['input_file']).read().decode() == 'Junk'
    assert gridfs.get(record['calculations']['test']['output_file']).read().decode() == 'More Junk'

