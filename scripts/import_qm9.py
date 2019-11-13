"""Read in the QM9 database into the QCFractal instance"""
from qcportal.client import FractalClient
from qcportal import Molecule
from edw.actions import mongo
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse

# Parse input arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('qm9_path', help='Path to the QM9 data file')
arg_parser.add_argument('password', help='Password for the service')
arg_parser.add_argument('--address', help='Address to QCFractal service', default='localhost:7874', type=str)
args = arg_parser.parse_args()

# Make the FractalClient
client = FractalClient(args.address, verify=False, username='user', password=args.password)

# Load in the data
data = pd.read_json(args.qm9_path, lines=True)

# Put it in the database
for chunk in tqdm(np.array_split(data, len(data) // 1000)):
    mol_chunks = []
    for rid, mol in chunk.iterrows():
        # Make a Molecule
        inchi_key = mongo.compute_inchi_key(mol['smiles_0'])
        mol = Molecule.from_data(mol['xyz'], dtype='xyz', identifiers={'inchikey': inchi_key})

        # Add molecule to QCFractal
        mol_chunks.append(mol)
    client.add_molecules(mol_chunks)
