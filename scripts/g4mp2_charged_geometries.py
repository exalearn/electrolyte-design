from parsl.executors import ThreadPoolExecutor, HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname
from parsl.config import Config
from concurrent.futures import as_completed
from pymongo import MongoClient
from edw.actions import mongo, gaussian
from edw.parsl import apps
from gridfs import GridFS
from tqdm import tqdm
import argparse
import parsl
import os

# Define how to launch Gaussian
gaussian_cmd = ['g16']

# Parse user arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dry-run', help='Whether to just get workload',
                        action='store_true', default=False)
arg_parser.add_argument('--mongo-host', help='Hostname for the MongoDB',
                        default='localhost', type=str)
arg_parser.add_argument('--limit', help='Maximum number of molecules to run',
                        default=0, type=int)
arg_parser.add_argument('--request-size', help='Number of nodes to request per allocation',
                        default=1, type=int)
args = arg_parser.parse_args()

# Make a executor
config = Config(
    app_cache=False,
    retries=2,
    executors=[
        HighThroughputExecutor(
            label='bebop_gaussian',
            address=address_by_hostname(),
            max_workers=1,
            provider=SlurmProvider(
                partition='bdwall',
                launcher=SrunLauncher(),
                nodes_per_block=args.request_size,
                init_blocks=0,
                min_blocks=0,
                max_blocks=20,
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

# Get the workload
query = {
    'geometry.oxidized': {'$exists': True},
    'geometry.reduced': {'$exists': True}
}
projection = ['inchi_key', 'geometry.oxidized', 'geometry.reduced']
n_records = collection.count_documents(query)
cursor = collection.find(query, projection, limit=args.limit)

if args.dry_run:
    print(f'Found {n_records} records')
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
        # Get the geometry for this charge state
        name = 'oxidized' if charge == 1 else 'reduced'
        xyz = record['geometry'][name]

        # Submit the Gaussian jobs
        run_dir = os.path.join('gaussian-run', f'{rid}_{name}_g4mp2')
        os.makedirs(run_dir, exist_ok=True)
        input_file = gaussian.make_input_file(xyz, functional='g4mp2', basis_set='', charge=charge)
        calc = apps.run_gaussian(gaussian_cmd, input_file, run_dir)
        data = apps.match_future_with_inputs((rid, charge), calc)
        jobs.append(data)

for j in tqdm(as_completed(jobs), desc='Completed', total=len(jobs)):
    tag, result = j.result()

    # Get the inchi key and charge
    inchi_key, charge = tag

    # Store the calculation results
    name = f'oxidized_g4mp2' if charge == 1 else f'reduced_g4mp2'
    mongo.add_calculation(collection, gridfs, inchi_key, name,
                          result['input_file'], result['output_file'], 'Gaussian',
                          result['successful'])

