"""Submit a batch of B3LYP calculations to run with Psi4"""
from qcportal.client import FractalClient
import argparse

# Parse input arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('password', help='Password for the service')
arg_parser.add_argument('--address', help='Address to QCFractal service', default='localhost:7874', type=str)
args = arg_parser.parse_args()

# Maximum number ot submit

# Make the FractalClient
client = FractalClient(args.address, verify=False, username='user', password=args.password)
mols = client.query_molecules(limit=1000)

# Put it in the database
comp = client.add_compute(program='psi4', driver='energy', method='b3lyp',
                          basis='6-31G(2df,p)', molecule=mols)
