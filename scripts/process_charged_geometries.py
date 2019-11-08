"""Makes sure geometries converged correctly.
If not, resubmits the optimization"""
from parsl.executors import ThreadPoolExecutor, HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname
from parsl.config import Config
from concurrent.futures import as_completed
from edw.actions import mongo, gaussian
from edw.parsl import apps
from pymongo import MongoClient
from gridfs import GridFS
from tqdm import tqdm
import parsl
import json

# Define how to launch Gaussian
gaussian_cmd = ['g16']


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
                nodes_per_block=1,
                init_blocks=0,
                min_blocks=0,
                max_blocks=8,
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


# Connect to Mongo and GridFS
client = MongoClient('localhost')
collection = mongo.initialize_collection(client)
gridfs = GridFS(client.get_database('jcesr'))

# Query for completed relaxation
query = {
    'calculation.oxidized_b3lyp': {'$exists': True},
    'calculation.reduced_b3lyp': {'$exists': True}
}
n_records = collection.count_documents(query)
cursor = collection.find(query)
print(f'Found {n_records} records with complete b3lyp calculations for both states')

# Parse the output
failures = []
jobs = []
for record in tqdm(cursor, total=n_records):
    for state in ['reduced', 'oxidized']:
        # Get the smiles
        smiles = record['identifiers']['smiles']
        inchi_key = record['inchi_key']
        tag = ":".join((inchi_key, state))

        # Determine the charge
        charge = 1 if state == 'oxidized' else -1

        # Get the output file
        calc_name = f'{state}_b3lyp'
        output_file = record['calculation'][calc_name]['output_file']
        output_file = gridfs.get(output_file)
        output_file = output_file.read().decode()

        # Check if the relaxation completed successfully
        converged, new_structure = gaussian.validate_relaxation(output_file)
        if new_structure is None:
            raise ValueError()

        # If converged, store the result. We're done!
        if converged:
            collection.update_one({'inchi_key': inchi_key},
                                  {'$set':
                                      {f'calculation.{calc_name}': {
                                          'validated': True
                                  }}})
            mongo.add_geometry(collection, inchi_key, state, new_structure)
        else:
            failures.append(inchi_key)
            charged_calc = apps.relax_gaussian(inchi_key, new_structure,
                                               gaussian_cmd, charge=charge,
                                               functional='B3LYP')
            data = apps.match_future_with_inputs((inchi_key, charge), charged_calc)
            jobs.append(data)

# Summarize and exit if no work to do!
print(f'{len(failures)} failures detected')

if len(failures) == 0:
    exit()

# Write the failures to disk
with open('failures.json', 'w') as fp:
    json.dump(failures, fp, indent=2)

# Wait until the complete
for j in tqdm(as_completed(jobs), desc='Completed', total=len(jobs)):
    tag, result = j.result()

    # Get the inchi key and charge
    inchi_key, charge = tag

    # Store the calculation results
    name = 'oxidized_b3lyp' if charge == 1 else 'reduced_b3lyp'
    mongo.add_calculation(collection, gridfs, inchi_key, name,
                          result['input_file'], result['output_file'], 'Gaussian',
                          result['successful'])
