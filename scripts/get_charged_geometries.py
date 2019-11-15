"""Run geometry relaxations for the first 1000 molecules in our database"""
from qcportal.client import FractalClient
from qcportal import Molecule
import argparse

# Parse input arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('password', help='Password for the service')
arg_parser.add_argument('--address', help='Address to QCFractal service', default='localhost:7874', type=str)
args = arg_parser.parse_args()

# Make the FractalClient
client = FractalClient(args.address, verify=False, username='user', password=args.password)
mols = client.query_molecules(limit=1000)

# Make a charged version of each molecule and submit the molecule and charge
for name, charge in zip(['reduced', 'oxidized'], [-1, 1]):
    new_mols = []
    for mol in mols:
        new_mol = Molecule.from_data(
            mol.to_string('xyz'),
            molecular_charge=charge,
            name=f'{mol.name}_{name}'
        )
        new_mols.append(new_mol)

    # Add the molecules to server
    client.add_molecules(new_mols)

    # Submit optimization records
    spec = {
        "keywords": None,
        "qc_spec": {
            "driver": "gradient",
            "method": "b3lyp",
            "basis": "6-31G(2df,p)",
            "program": "psi4"
        },
    }
    client.add_procedure("optimization", "geometric", spec, new_mols)
