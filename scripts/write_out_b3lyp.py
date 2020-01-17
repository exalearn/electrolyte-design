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
    'calculation.oxidized_b3lyp': {'$exists': True},
    'calculation.oxidized_b3lyp': {'$exists': True}
}
total_count = collection.count_documents(query)
print(f'Found {total_count} completed calculations')
cursor = collection.find(query)

# Extract the G4MP2 results
did_not_complete = []
for record in tqdm(cursor, total=total_count, desc='Extracting energies'):
    rid = record['inchi_key']
    for calc_name in ['oxidized_b3lyp', 'reduced_b3lyp']:
        # Determine the charge state
        charge_state = calc_name.split("_")[0]
        if calc_name not in record['calculation']:
            did_not_complete.append(rid)
            continue

        # Read in the output file
        output_text = gridfs.get(record['calculation'][calc_name]['output_file']).read().decode()

        # Get the B3LYP energy
        b3lyp_match = re.findall(r'Sum of electronic and zero-point Energies=           (?P<energy>-[\d\.]+)', output_text)
        if len(b3lyp_match) == 0:
            did_not_complete.append(rid)
            continue
        b3lyp_energy = float(re.sub(r'\s', '', b3lyp_match[-1]))
        mongo.add_property(collection, rid, f'u0_{charge_state}', 'b3lyp', b3lyp_energy)

print(f'Errors in {len(did_not_complete)} calculations')

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

pd.DataFrame(output).to_csv('b3lyp_results.csv', index=False)

