"""Read in the QM9 database into the MongoDB"""
from pymongo import MongoClient
from edw.actions import mongo
from gridfs import GridFS
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse

# Parse input arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('qm9_path', help='Path to the QM9 data file')
arg_parser.add_argument('--port', help='MongoDB port', default=None, type=int)
args = arg_parser.parse_args()

# Make the MongoClient
client = MongoClient('localhost', port=args.port)
collection = mongo.initialize_collection(client)
gridfs = GridFS(client.get_database('jcesr'))

# Load in the data
data = pd.read_json(args.qm9_path, lines=True)

# Put it in the database
for rid, mol in tqdm(data.iterrows(), total=len(data)):
    # Add a record
    mongo.add_molecule(collection, mol['smiles_0'], 'From QM9')
    inchi_key = mongo.compute_inchi_key(mol['smiles_0'])

    # Add in the relaxed geometry
    mongo.add_geometry(collection, inchi_key, 'neutral', mol['xyz'])

    # Add in some B3LYP properties
    for prop in ['u0', 'zpe', 'homo', 'lumo']:
        mongo.add_property(collection, inchi_key, prop, 'b3lyp', mol[prop])

    # Add in the G4MP2 energy
    mongo.add_property(collection, inchi_key, 'u0', 'g4mp2', mol['g4mp2_0k'])

    # If the molecule is in the holdout set, add subset information
    if mol['in_holdout']:
        collection.update_one(filter={'inchi_key': inchi_key},
                              update={'$set': {'subset': ['holdout']}})
