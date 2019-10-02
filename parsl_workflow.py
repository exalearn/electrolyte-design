from concurrent.futures import as_completed
from edw.parsl import generate_conformers, relax_structure
from parsl.configs.htex_local import config
import parsl


# Set up parsl
parsl.load(config)

# Workload
test_smiles = ['C'] * 20

# Generate the conformers
jobs = []
for r in as_completed(generate_conformers(test_smiles)):
    smiles, confs = r.result()

    # Relax the conformers for each molecule with B3LYP
    for i, c in enumerate(confs):
        jobs.append(relax_structure(f'{smiles}-{i}', c))

# Print the jobs as they complete
for j in as_completed(jobs):
    print(j.result())
