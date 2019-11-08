"""Run geometry optimizations for oxidized and reduced molecules

Gathers work by finding structures in the MongoDB without these geometries.
Submits Gaussian calculations via Parsl, checks whether results are converged,
and continues to resubmit structures until they are completely relaxed.
Completed structures are stored in the MongoDB.
"""

from parsl.executors import ThreadPoolExecutor, HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname
from parsl.config import Config
from logging.handlers import RotatingFileHandler
from concurrent.futures import as_completed
from pymongo import MongoClient
from edw.actions import mongo
from edw.parsl import apps, chains
from gridfs import GridFS
from tqdm import tqdm
import argparse
import logging
import parsl
import os

# Setup the logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    handlers=RotatingFileHandler(
                        filename=os.path.join('logs', 'get_charged_geometries.log'),
                        maxBytes=1024 * 1024 * 16,
                        backupCount=4
                    ))
logger = logging.getLogger(__name__)

# Parse user arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dry-run', help='Get the workfload but do not run tasks',
                        action='store_true', default=False)
arg_parser.add_argument('--mongo-host', help='Hostname for the MongoDB',
                        default='localhost', type=str)
arg_parser.add_argument('--limit', help='Maximum number of molecules to run',
                        default=0, type=int)
args = arg_parser.parse_args()

# Define how to launch Gaussian
gaussian_cmd = ['g16']

# Define the Parsl execution environment
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
                max_blocks=10,
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

# Set up the query for molecules without
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
    inchi_key = record['inchi_key']
    starting_geometry = record['geometry']['neutral']
    for charge, geom_name, calc_name in zip([-1, 1],
                                            ('reduced', 'oxidized'),
                                            ('reduced_b3lyp', 'oxidized_b3lyp')):

        charged_calc = chains.robust_relaxation(inchi_key, calc_name, gaussian_cmd,
                                                starting_geometry, geom_name,
                                                collection, gridfs, charge=charge)
        jobs.append(charged_calc)


# Wait until all tasks complete
counter = tqdm(total=len(jobs), desc='Completed')
while len(jobs) > 0:
    # Wait for a task to complete
    job = next(as_completed(jobs))
    logger.info(f'Task {job.tid} completed')

    # Remove job from the current queue
    jobs.remove(job)

    # Get the result
    result = job.result()

    # If it is None, the job is complete. Otherwise, add the current result
    #  back into the pool of "yet to be completed Parsl jobs"
    if result is not None:
        logger.info(f'Task {job.tid} created a new task {result.tid}')
        jobs.append(result)  # We are not done yet
    else:
        counter.update()  # Increments the status bar
