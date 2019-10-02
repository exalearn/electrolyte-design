from parsl.executors.threads import ThreadPoolExecutor
from parsl.configs.htex_local import config
from concurrent.futures import as_completed
from edw.parsl import apps
from tqdm import tqdm
import parsl
import json

# Define how to deal with NWCHem
nwchem = ['mpirun', '-n', '1', 'nwchem']

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
test_smiles = ['C'] * 20

# Assemble the workflow
jobs = []
for smiles in tqdm(test_smiles, desc='Submitted'):
    confs = apps.smiles_to_conformers(smiles, 16)
    relaxed_confs = apps.relax_conformers(confs, nwchem)
    data = apps.collect_conformers(smiles, relaxed_confs)
    jobs.append(data)

for j in tqdm(as_completed(jobs), desc='Completed', total=len(jobs)):
    continue
