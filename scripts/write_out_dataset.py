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
arg_parser.add_argument('--restart', action='store_true', help='Delete failed calculations so they can be restarted')
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
did_not_complete = []
for record in tqdm(cursor, total=total_count, desc='Extracting energies'):
    rid = record['inchi_key']
    for calc_name in ['oxidized_g4mp2', 'reduced_g4mp2']:
        # Determine the charge state
        charge_state = calc_name.split("_")[0]
        if calc_name not in record['calculation']:
            raise ValueError(f'Missing {calc_name}')

        # Read in the output file
        output_text = gridfs.get(record['calculation'][calc_name]['output_file']).read().decode()

        # Get the B3LYP energy
        b3lyp_match = re.search(r'E\(UB3LYP\) = +([-\d.]+)', output_text)
        if b3lyp_match is None:
            did_not_complete.append((rid, calc_name))
            continue
        b3lyp_energy = float(re.sub(r'\s', '', b3lyp_match.group(1)))
        mongo.add_property(collection, rid, f'u0_{charge_state}', 'b3lyp', b3lyp_energy)

        # Read the ZPE
        zpe_match = re.search(r'E\(ZPE\)= +([-\d.]+)', output_text)
        if zpe_match is None:
            did_not_complete.append((rid, calc_name))
            continue
        zpe = float(re.sub(r'\s', '', zpe_match.group(1)))
        mongo.add_property(collection, rid, f'zpe_{charge_state}', 'g4mp2', zpe)


        # Read the G4MP2 energy
        g4mp2_match = re.search(r'G4MP2=([-\d.\s]+)', output_text)
        if g4mp2_match is None:
            did_not_complete.append((rid, calc_name))
            continue
        g4mp2_energy = float(re.sub(r'\s', '', g4mp2_match.group(1)))
        mongo.add_property(collection, rid, f'u0_{charge_state}', 'g4mp2', g4mp2_energy)


print(f'Errors in {len(did_not_complete)} calculations')

# Restart them if need be
if args.restart:
    for inchi, calc_name in tqdm(did_not_complete, desc='Restarting'):
        # Get the failed record
        record = collection.find_one({'inchi_key': inchi})

        # Delete the files
        calc_info = record['calculation'][calc_name]
        for f in ['input_file', 'output_file']:
            f = calc_info[f]
            gridfs.delete(f)

        # Unset the record
        collection.update_one(filter={'inchi_key': inchi}, 
                update={'$unset': {f'calculation.{calc_name}': ''}})


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

    # Write out molecule details
    for key, value in record['details'].items():
        summary[key] = value

    # Store the energies
    for property, values in record['property'].items():
        for level, value in values.items():
            summary[f'{property}.{level}'] = value

    output.append(summary)

pd.DataFrame(output).to_csv('g4mp2_results.csv', index=False)

