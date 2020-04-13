"""Add molecule size to the records"""
from pymongo import MongoClient
from edw.actions import mongo
from gridfs import GridFS
from rdkit import Chem
from tqdm import tqdm
import argparse
import logging
import os

# Parse user arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--mongo-host', help='Hostname for the MongoDB',
                        default='localhost', type=str)
args = arg_parser.parse_args()

# Connect to MongoDB
client = MongoClient(args.mongo_host)
gridfs = GridFS(client.get_database('jcesr'))
collection = mongo.initialize_collection(client)

# Set up the query for molecules without
query = {}
projection = ['inchi_key', 'identifiers.smiles']
n_records = collection.count_documents(query)
print(f'{n_records} records to process')

# Compute the atom count for each of the molecules
cursor = collection.find(query, projection)

# Assemble the workflow
jobs = []
for record in tqdm(cursor, desc='Processed', total=n_records):
    inchi_key = record['inchi_key']
    smiles = record['identifiers']['smiles']
    n_atoms = Chem.MolFromSmiles(smiles).GetNumAtoms()
    collection.update_one(filter={'inchi_key': inchi_key},
                          update={'$set': {'details.n_heavy_atoms': n_atoms}})

