from moldesign.simulate.qcfractal import GeometryDataset
from tqdm import tqdm
import pandas as pd
import argparse

# Parse input arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("molecules", nargs="+", help="List of molecules as SMILES strings")
args = arg_parser.parse_args()

# Add these molecules to the database
geom = GeometryDataset('Electrolyte Geometry XTB', 'xtb')
total_added = 0
for smiles in tqdm(args.molecules):
    try:
        was_added = geom.add_molecule_from_smiles(smiles)
    except BaseException:
        was_added = False
    if was_added:
        total_added += 1
print(f'Added {total_added} molecules')

# Start some XTB calculations
n_started = geom.start_compute()
print(f'Started {n_started} calculations')
