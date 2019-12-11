from pymongo import MongoClient
from edw.actions import mongo
from gridfs import GridFS
from tqdm import tqdm
import pandas as pd
import argparse
import re

# Parse input arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--port', help='MongoDB port', default=None, type=int)
args = arg_parser.parse_args()

# Make the MongoClient
client = MongoClient('localhost', port=args.port)
collection = mongo.initialize_collection(client)
gridfs = GridFS(client.get_database('jcesr'))

# Query for completed calculations
query = {
    'calculation.oxidized_g4mp2': {'$exists': True},
    'calculation.reduced_g4mp2': {'$exists': True}
}
total_count = collection.count_documents(query)
print(f'Found {total_count} completed calculations')
cursor = collection.find(query)

# Extract the G4MP2 results
for record in tqdm(cursor, total=total_count, desc='Extracting energies'):
    rid = record['inchi_key']
    for calc_name in ['oxidized_g4mp2', 'reduced_g4mp2']:
        # Determine the charge state
        charge_state = calc_name.split("_")[0]
        if calc_name not in record['calculation']:
            raise ValueError(f'Missing {calc_name}')

        # Access the archive string at the end
        output_text = gridfs.get(record['calculation'][calc_name]['output_file']).read().decode()
        g4mp2_match = re.search(r'G4MP2=([-\d.\s]+)', output_text)
        g4mp2_energy = float(re.sub(r'\s', '', g4mp2_match.group(1)))

        # Also get the B3LYP energy
        b3lyp_match = re.search(r'E\(UB3LYP\) = +([-\d.]+)', output_text)
        b3lyp_energy = float(re.sub(r'\s', '', b3lyp_match.group(1)))

        # Add the energy to the database
        mongo.add_property(collection, rid, f'u0_{charge_state}', 'g4mp2', g4mp2_energy)
        mongo.add_property(collection, rid, f'u0_{charge_state}', 'b3lyp', b3lyp_energy)

# Loop through and dump out the energies
cursor = collection.find(query)
output = []
for record in tqdm(cursor, total=total_count, desc='Saving results'):
    # Store basic information
    summary = {
        'inchi_key': record['inchi_key'],
        'smiles': record['identifiers']['smiles']
    }

    # Store the geometries
    for key, xyz in record['geometry'].items():
        summary[f'xyz_{key}'] = xyz

    # Store the energies
    for property, values in record['property'].items():
        for level, value in values.items():
            summary[f'{property}.{level}'] = value

    output.append(summary)

pd.DataFrame(output).to_csv('g4mp2_results.csv', index=False)

