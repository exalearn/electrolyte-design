from parsl.executors import ThreadPoolExecutor, HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname
from parsl.config import Config
from concurrent.futures import as_completed
from edw.parsl import apps
from tqdm import tqdm
import pandas as pd
import parsl
import json

# Define how to deal with Gaussian
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
                nodes_per_block=1,
                init_blocks=1,
                max_blocks=1,
                worker_init='''
module load gaussian/16-a.03
export GAUSS_SCRDIR=/scratch
export GAUSS_WDEF="$(scontrol show hostname $SLURM_JOB_NODELIST | paste -d, -s)"
export GAUSS_CDEF=0-35
export GAUSS_MDEF=100GB
export GAUSS_SDEF=ssh
export GAUSS_LFLAGS="-vv"''',
               walltime="0:15:00"
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

# Workload
data = pd.read_json('qm9.jsonld', lines=True)
data = data.sample(1)

# Assemble the workflow
jobs = []
for rid, row in tqdm(data.iterrows(), desc='Submitted'):
    for charge in [-1, 1]:
        charged_calc = apps.relax_gaussian(str(rid), row['xyz'], gaussian_cmd,
                                           charge=charge, functional='B3LYP')
        data = apps.match_future_with_inputs((rid, charge), charged_calc)
        jobs.append(data)

for j in tqdm(as_completed(jobs), desc='Completed', total=len(jobs)):
    print(j.result())
