"""Add some molecules"""
from qcportal.collections import OptimizationDataset
from qcportal.client import FractalClient
from qcportal import Molecule
import pandas as pd
import argparse

# Hard-coded stuff
coll_name = 'NWChem+Geometric Relaxation Test'
def create_specs(client: FractalClient):
    """Make the desired specifications for our tests
    
    Requires using keyword sets, hence the need for the client.
    
    Args:
        client (FractalClient): Client used for making found_specs
    """

    # Make the keyword sets
    fine_kwds, xfine_kwds = client.add_keywords([
        {'values': {
            'dft__convergence__energy': '1e-7',
            'dft__convergence__density': '1e-6',
            'dft__convergence__gradient': '5e-5',
            'dft__grid': 'fine'
        }, 'comments': 'Tight convergence settings for NWChem'},
        {'values': {
            'dft__convergence__energy': '1e-7',
            'dft__convergence__density': '1e-6',
            'dft__convergence__gradient': '5e-5',
            'dft__grid': 'xfine'
        }, 'comments': 'Very tight convergence settings for NWChem'}
    ])

    # Return the specifications
    return [{
        'name': 'small_basis',
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
    }, {
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
    }, {
        'name': 'default_fine',
        'description': 'Geometric + NWChem/B3LYP/6-31g(2df,p) with fine convergence.',
        'optimization_spec': {
            'program': 'geometric',
            'keywords': None
        }, 'qc_spec': {
            'driver': 'gradient',
            'method': 'b3lyp',
            'basis': '6-31g(2df,p)',
            'program': 'nwchem',
            'keywords': fine_kwds
        }
    },{
        'name': 'default_xfine',
        'description': 'Geometric + NWChem/B3LYP/6-31g(2df,p) with xfine convergence.',
        'optimization_spec': {
            'program': 'geometric',
            'keywords': None
        }, 'qc_spec': {
            'driver': 'gradient',
            'method': 'b3lyp',
            'basis': '6-31g(2df,p)',
            'program': 'nwchem',
            'keywords': xfine_kwds
        }
    },{
        'name': 'small_basis_fine',
        'description': 'Geometric + NWChem/B3LYP/3-21g with fine convergence.',
        'optimization_spec': {
            'program': 'geometric',
            'keywords': None
        }, 'qc_spec': {
            'driver': 'gradient',
            'method': 'b3lyp',
            'basis': '3-21g',
            'program': 'nwchem',
            'keywords': fine_kwds
    }}]

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
try:
    coll = OptimizationDataset.from_server(name=coll_name, client=client)
except KeyError:
    coll = OptimizationDataset(name=coll_name, client=client)
    coll.save()

# Make sure it has the right calculation specs
found_specs = coll.list_specifications(description=False)
print(f'Found the following specifications: {found_specs}')
desired_specs = create_specs(client)
for spec in desired_specs:
    if spec["name"] not in found_specs:
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
for spec in desired_specs:
    n_started = coll.compute(spec['name'])
    print(f'Started {n_started} {spec["name"]} calculations')
