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
    # Get the record ID
    inchi_key = mongo.compute_inchi_key(mol['smiles_0'])

    # Add in some B3LYP properties we missed earlier
    for prop in ['g']:
        mongo.add_property(collection, inchi_key, prop, 'b3lyp', mol[prop])

    # Add in the solvation properties
    for col, name in zip(['sol_water', 'sol_ethanol', 'sol_dmso', 'sol_acn', 'sol_acetone'],
            ['water', 'ethanol', 'dimethylsulfoxide', 'acetonitrile', 'acetone']):
        mongo.add_property(collection, inchi_key, 'solvent_neutral', f'B3LYP_{name}', mol[col])
 
    # Add in the GDB identifier
    collection.update_one(filter={'inchi_key': inchi_key},
                          update={'$set': {
                              'identifiers.gdb': mol['filename'],
                              'details.n_atoms': mol['n_atom'],
                              'details.n_heavy_atoms': mol['n_heavy_atoms'],
                              'details.n_electrons': mol['n_electrons']
                              }})
