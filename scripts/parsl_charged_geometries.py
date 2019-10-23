from parsl.executors import ThreadPoolExecutor, HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname
from parsl.config import Config
from concurrent.futures import as_completed
from pymongo import MongoClient
from edw.actions import mongo
from edw.parsl import apps
from gridfs import GridFS
from tqdm import tqdm
import pandas as pd
import argparse
import parsl
import json


# Parse user arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dry-run', help='Whether to just get workload',
                        action='store_true', default=False)
arg_parser.add_argument('--mongo-host', help='Hostname for the MongoDB',
                        default='localhost', type=str)
arg_parser.add_argument('--limit', help='Maximum number of molecules to run',
                        default=0, type=int)
args = arg_parser.parse_args()

# Define how to launch Gaussian
gaussian_cmd = ['g16']

# Make a executor
config = Config(
    executors=[
        HighThroughputExecutor(
            label='bebop_gaussian',
            address=address_by_hostname(),
            max_workers=1,
            provider=SlurmProvider(
                partition='bdwall',
                launcher=SrunLauncher(),
                nodes_per_block=4,
                init_blocks=1,
                min_blocks=0,
                max_blocks=1,
                worker_init='''
module load gaussian/16-a.03
export GAUSS_SCRDIR=/scratch
export GAUSS_WDEF="$(scontrol show hostname $SLURM_JOB_NODELIST | paste -d, -s)"
export GAUSS_CDEF=0-35
export GAUSS_MDEF=100GB
export GAUSS_SDEF=ssh
export GAUSS_LFLAGS="-vv"''',
               walltime="12:00:00"
            )
        )
    ]
)

# Connect to MongoDB
client = MongoClient(args.mongo_host)
gridfs = GridFS(client.get_database('jcesr'))
collection = mongo.initialize_collection(client)

# Set up the query
query = {
    'subset': 'holdout',
    '$or': [{'calculation.oxidized_b3lyp': {'$exists': False}},
            {'calculation.reduced_b3lyp': {'$exists': False}}]
}
projection = ['inchi_key', 'geometry.neutral']
n_records = collection.count_documents(query)
print(f'{n_records} records left to process')

# Get the workload
if args.limit > 0:
    n_records = min(n_records, args.limit)
    print(f'Only running {n_records} of them')
cursor = collection.find(query, projection, limit=args.limit)

if args.dry_run:
    next_record = cursor.next()
    next_record.pop('_id')
    print(f'First record:\n{next_record}')
    exit()

# Add local threads to the config
config.executors.append(ThreadPoolExecutor(label='local_threads'))

# Set up parsl
parsl.load(config)

# Mark which apps cannot use the local_threads apps
worker_execs = [x.label for x in config.executors if x.label != 'local_threads']
for app_name in apps.__all__:
    app = getattr(apps, app_name)
    if app.executors == 'all':
        app.executors = worker_execs
        print(f'Assigned app {app_name} to executors: {app.executors}')


# Assemble the workflow
jobs = []
for record in tqdm(cursor, desc='Submitted', total=n_records):
    rid = record['inchi_key']
    for charge in [-1, 1]:
        charged_calc = apps.relax_gaussian(str(rid), record['geometry']['neutral'],
                                           gaussian_cmd, charge=charge, functional='B3LYP')
        data = apps.match_future_with_inputs((rid, charge), charged_calc)
        jobs.append(data)

with open('qm9-charged.jsonld', 'a') as fp:
    for j in tqdm(as_completed(jobs), desc='Completed', total=len(jobs)):
        tag, result = j.result()
        
        # Get the inchi key and charge
        inchi_key, charge = tag

        # Store the calculation results
        name = 'oxidized_b3lyp' if charge == 1 else 'reduced_b3lyp'
        mongo.add_calculation(collection, gridfs, inchi_key, name,
                              result['input_file'], result['output_file'], 'Gaussian',
                              result['successful'])
