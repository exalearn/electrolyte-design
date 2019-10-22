"""Actions related to database storage"""

import sys
from typing import Any
from subprocess import Popen

import pymongo
from gridfs import GridFS
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection

from pymongo.errors import DuplicateKeyError
from pymongo.results import UpdateResult

from .geometry import compute_inchi_key


def launch_mongodb(database_path: str, port: int) -> Popen:
    """Launch a MongoDB server

    Tries to find an open port automatically

    Args:
        database_path (str): Path to the database
        port (int): Port number to use
    Returns:
        - (Popen) MongoDB process
    """

    # Launch mongod
    proc = Popen(['mongod', '--port', str(port), '--dbpath', database_path],
                 stderr=sys.stderr, stdout=sys.stdout)

    return proc


def initialize_collection(client: MongoClient,
                          database: str = "jcesr",
                          collection: str = "electrolytes") -> Collection:
    """Make a new collection of the electrolyte data or get a pointer to an existing one

    Args:
        client (MongoClient): Client for Mongo database
        database (str): Name of the database
        collection (str): Name of the collection
    Returns:
        (str) Pointer to that collection
    """

    db = client.get_database(database)
    col = db.get_collection(collection)

    # Add indexing options if collection does not already exist
    if collection not in db.list_collection_names():
        col.create_index([('inchi_key', pymongo.ASCENDING)], unique=True)

    return col


def add_molecule(collection: Collection, smiles: str, notes: str = '') -> bool:
    """Add a molecule to the database

    Initializes a record

    Args:
        collection (Collection): Collection in which to store the molecule
        smiles (str): SMILES string for the molecule
        notes (str): Reason why this molecule was added
    Returns:
        (bool) Whether the molecule was added successfully
    """

    # Compute the inchi key
    inchi_key = compute_inchi_key(smiles)

    now = datetime.now()
    try:
        collection.insert_one({'inchi_key': inchi_key, 'notes': notes,
                               'identifiers': {'smiles': smiles},
                               'created': now, 'updated': now})
    except DuplicateKeyError:
        return False
    return True


def add_calculation(collection: Collection, gridfs: GridFS,
                    inchi_key: str, name: str,
                    input_file: str, output_file: str,
                    code: str, completed: bool = True) -> UpdateResult:
    """Append a calculation to a certain record

    Does not parse the output record

    Args:
        collection (Collection): Collection in which to store the calculation results
        gridfs (GridFS): Store in which to place the output files
        inchi_key (str): Identifier for the molecule
        name (str): Name of the calculation type
        input_file (str): Content of the input file
        output_file (str): Content of the output file
        code (str): Name of software used to perform the calculation
    """

    # Upload the files to GridFS
    infile_id = gridfs.put(input_file.encode(), filename=f'{inchi_key}-{name}.in')
    outfile_id = gridfs.put(output_file.encode(), filename=f'{inchi_key}-{name}.out')

    # Make the update record
    update_cmd = {
        '$set': {
            f'calculations.{name}': {
                'input_file': infile_id,
                'output_file': outfile_id,
                'code': code,
                'completed': completed
            }
        }
    }

    # Append the calculation
    return collection.update_one(filter={'inchi_key': inchi_key}, update=update_cmd)


def add_geometry(collection: Collection, inchi_key: str, name: str, xyz: str):
    """Add a geometry to a molecule record

    Args:
        collection (Collection): Collection in which to store record
        inchi_key (str): Identifier of the molecule
        name (str): Name of the geometry
        xyz (str): XYZ-format description of the geometry
    """

    update_cmd = {
        "$set": {
            f'geometries.{name}': xyz
        }
    }
    collection.update_one(filter={'inchi_key': inchi_key}, update=update_cmd)


def add_property(collection: Collection, inchi_key: str, name: str, level: str, value: Any):
    """Add a derived property for a molecule

    Args:
        collection (Collection): Collection in which to store record
        inchi_key (str): Identifier of the molecule
        name (str): Name of the property
        level (str): Name of the fidelity level
        value: Property value to store
    """

    update_cmd = {
        "$set": {
            f'properties.{name}.{level}': value
        }
    }
    collection.update_one(filter={'inchi_key': inchi_key}, update=update_cmd)
