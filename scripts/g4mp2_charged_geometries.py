from parsl.executors import ThreadPoolExecutor, HighThroughputExecutor
from parsl.providers import CobaltProvider
from parsl.launchers import SimpleLauncher
from parsl.addresses import address_by_hostname
from parsl.config import Config
from concurrent.futures import as_completed
from pymongo import MongoClient
from edw.actions import mongo, nwchem
from edw.parsl import apps
from gridfs import GridFS
from tqdm import tqdm
import argparse
import parsl


# Parse user arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dry-run', help='Whether to just get workload',
                        action='store_true', default=False)
arg_parser.add_argument('--mongo-host', help='Hostname for the MongoDB',
                        default='localhost', type=str)
arg_parser.add_argument('--limit', help='Maximum number of molecules to run',
                        default=0, type=int)
arg_parser.add_argument('--nodes-per-job', help='Number of nodes per nwchem job',
                        default=8, type=int)
arg_parser.add_argument('--request-size', help='Number of nodes to request per allocation',
                        default=8, type=int)
arg_parser.add_argument('--jobs')
args = arg_parser.parse_args()

# Define how to launch NWChem
ranks_per_node = 16
threads_per_rank = 4
threads_per_core = 1
nwchem_cmd = ['aprun', '-n', f'{args.nodes_per_job * ranks_per_node}',
              '-N', f'{ranks_per_node}',
              '-d', f'{threads_per_rank}',
              '-j', '1',
              '-cc', 'depth',
              '--env', f'OMP_NUM_THREADS={threads_per_rank}',
              '--env', f'MKL_NUM_THREADS={threads_per_rank}',
              '-j', f'{threads_per_core}'
              '/soft/applications/nwchem/6.8/bin/nwchem']

# Determine the number of workers per executor
max_workers = args.request_size / args.nodes_per_job

# Make a executor
config = Config(
    executors=[
        HighThroughputExecutor(
            label='theta_aprun',
            address=address_by_hostname(),
            max_workers=max_workers,
            provider=CobaltProvider(
                queue='default',
                launcher=SimpleLauncher(),
                nodes_per_block=128,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                worker_init='''
module load atp
export MPICH_GNI_MAX_EAGER_MSG_SIZE=16384
export MPICH_GNI_MAX_VSHORT_MSG_SIZE=10000
export MPICH_GNI_MAX_EAGER_MSG_SIZE=131072
export MPICH_GNI_NUM_BUFS=300
export MPICH_GNI_NDREG_MAXSIZE=16777216
export MPICH_GNI_MBOX_PLACEMENT=nic
export MPICH_GNI_LMT_PATH=disabled
export COMEX_MAX_NB_OUTSTANDING=6
export LD_LIBRARY_PATH=/soft/compilers/intel/19.0.3.199/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64:$LD_LIBRARY_PATH''',
                walltime="0:30:00"
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
        name = 'oxidized' if charge == -1 else 'reduced'
        xyz = record['geometry'][name]

        # Make the input configuration
        #  TODO(wardlt): Hard-coding 3000mb for Theta memory maximum
        task_cfgs, input_cfgs = nwchem.generate_g4mp2_configs(charge, '3000 mb')

        # Submit the NWCHem jobs
        for level in task_cfgs:
            task_cfg = task_cfgs[level]
            input_cfg = input_cfgs[level]
            input_file = nwchem.make_input_file(xyz, task_cfg, input_cfg)
            calc = apps.run_nwchem(str(rid), xyz, nwchem_cmd=nwchem_cmd)
            data = apps.match_future_with_inputs((rid, level, charge), calc)
            jobs.append(data)

for j in tqdm(as_completed(jobs), desc='Completed', total=len(jobs)):
    tag, result = j.result()

    # Get the inchi key and charge
    inchi_key, level, charge = tag

    # Store the calculation results
    name = f'oxidized_{level}' if charge == 1 else f'reduced_{level}'
    mongo.add_calculation(collection, gridfs, inchi_key, name,
                          result['input_file'], result['output_file'], 'Gaussian',
                          result['successful'])
