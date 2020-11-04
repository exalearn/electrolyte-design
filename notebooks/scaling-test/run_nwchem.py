import argparse
import logging
import json
import os
logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')

from moldesign.simulate.specs import get_qcinput_specification
import qcelemental as qcel
import qcengine as qcng
import pandas as pd
import logging
import yaml
charge = 1
use_tce = True

# Make the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--driver", default="gradient", help="What QC driver to run")
parser.add_argument("--config", choices=["small_basis", "normal_basis"], help="Which NWChem config to use",
                    default="small_basis")
parser.add_argument("--sizes", nargs="+", help="Molecule sizes to run", default=[8, 40, 75, 126, 312], type=int)
parser.add_argument("--cores-per-rank", default=2, help="Number of cores per MPI rank", type=int)
parser.add_argument("--charge", default=0, help="Atomic charge")
args = parser.parse_args()

# Get the number of cores per node
with open('qcengine.yaml') as fp:
    qce_settings = yaml.load(fp)
proc_per_node = qce_settings['all']['ncores']

# Load in the molecules
mols = pd.read_csv("example_molecules.csv")
mols = mols[mols["num_electrons"].apply(lambda x: x in args.sizes)]
logging.info(f"Pulled {len(mols)} for testing with sizes: {mols['num_electrons'].to_list()}")

# Get the run config
spec, program = get_qcinput_specification(args.config)
logging.info(f"Pulled spec for '{args.config}'")

# Loop over the molecules
n_nodes = int(os.environ.get("COBALT_JOBSIZE", "1"))
for _, mol in mols.iterrows():
    # Parse the molecule
    mol_obj = qcel.models.Molecule.from_data(mol["xyz"], molecular_charge=args.charge)

    # Make the input
    input_obj = qcel.models.AtomicInput(molecule=mol_obj, driver=args.driver,
                                        **spec.dict(exclude={'driver'}))

    # Run the calculation
    ret = qcng.compute(input_obj, "nwchem", local_options={
        'scratch_directory': './scratch',
        'cores_per_rank': args.cores_per_rank,
        'nnodes': n_nodes}, raise_error=True)
    print(ret)
    with open('timings.json', 'a') as fp:
        print(json.dumps({
            'num_electrons': mol["num_electrons"],
            'inchi': mol["inchi"],
            'charge': charge,
            'nnodes': n_nodes,
            'cores_per_rank': args.cores_per_rank,
            'cores_per_node': proc_per_node,
            'walltime': ret.provenance.wall_time}), file=fp)
