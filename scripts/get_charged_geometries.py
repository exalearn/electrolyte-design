"""Add some molecules"""
from qcportal.collections import OptimizationDataset
from qcportal.client import FractalClient
from qcportal import Molecule
import pandas as pd
import argparse

# Parse input arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--limit', help='Number of molecules to add. -1 to add all', default=1, type=int)
arg_parser.add_argument('qm9_path', help='Path to the QM9 data file')
arg_parser.add_argument('password', help='Password for the service')
arg_parser.add_argument('--address', help='Address to QCFractal service',
                        default='localhost:7874', type=str)
args = arg_parser.parse_args()

# Make the FractalClient
client = FractalClient(args.address, verify=False, username='user', password=args.password)

# Assemble the dataset
coll = OptimizationDataset.from_server(name='NWCHem+Geometric Relaxation', client=client)
specs = coll.list_specifications(description=False)
print(f'Found the following specifications: {specs}')

if "tight" not in specs:
    print('Setting up computation settings...')

    # A full-accuracy version
    spec = {
        'name': 'default',
        'description': 'Geometric + NWChem/B3LYP/6-31g(2df,p).',
        'optimization_spec': {
            'program': 'geometric',
            'keywords': None
        }, 'qc_spec': {
            'driver': 'gradient',
            'method': 'b3lyp',
            'basis': '6-31g(2df,p)',
            'program': 'nwchem'
        }
    }
    coll.add_specification(**spec)
    
    # A full-accuracy version with tighter convergence
    tight_kwds = client.add_keywords([{'values': {
        'dft__convergence__energy': '1e-8',
        'dft__convergence__density': '1e-7',
        'dft__convergence__gradient': '5e-6'
    }, 'comments': 'Tight convergence settings for NWChem'}])[0]
    spec = {
        'name': 'tight',
        'description': 'Geometric + NWChem/B3LYP/6-31g(2df,p) with a tighter convergence threshold.',
        'optimization_spec': {
            'program': 'geometric',
            'keywords': None
        }, 'qc_spec': {
            'driver': 'gradient',
            'method': 'b3lyp',
            'basis': '6-31g(2df,p)',
            'program': 'nwchem',
            'keywords': tight_kwds
        }
    }

    coll.add_specification(**spec)

    # A reduced-accuracy version
    spec = {
        'name': 'low_fidelity',
        'description': 'Geometric + NWChem/B3LYP/3-21g.',
        'optimization_spec': {
            'program': 'geometric',
            'keywords': None
        }, 'qc_spec': {
            'driver': 'gradient',
            'method': 'b3lyp',
            'basis': '3-21g',
            'program': 'nwchem'
        }
    }

    coll.add_specification(**spec)
    coll.save()

# Get the current list of molecules
existing_mols = coll.df.index

# Load in QM9
data = pd.read_json(args.qm9_path, lines=True)
if args.limit > 0:
    data = data.head(args.limit)

# Add neutral charged versions of the molecules to the collection
mols_to_add = [Molecule.from_data(xyz, name=f'{smiles}_neutral') for xyz, smiles in zip(data['xyz'], data['smiles_0'])]

# Add the charged versions
for name, charge in zip(['reduced', 'oxidized'], [-1, 1]):
    for xyz, smiles in zip(data['xyz'], data['smiles_0']):
        new_mol = Molecule.from_data(
            xyz,
            molecular_charge=charge,
            name=f'{smiles}_{name}'
        )
        mols_to_add.append(new_mol)

# Submit the molecules to server
for mol in mols_to_add:
    if mol.name not in existing_mols:
        coll.add_entry(mol.name, mol, save=False)
coll.save()

# Trigger the calculations
for spec in specs:
    n_started = coll.compute(spec)
    print(f'Started {n_started} {spec} calculations')
