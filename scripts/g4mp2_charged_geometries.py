"""Run the G4MP2 calculations for a set of molecules"""
from qcelemental.models import Molecule
from qcfractal.interface.client import FractalClient
from qcfractal.interface.collections import Dataset
from qcfractal.interface.models import KeywordSet
from tqdm import tqdm
import pandas as pd
import argparse

# Hard-coded stuff
eng_coll_name = 'NWChem G4MP2 Charged'
hess_coll_name = 'NWChem G4MP2 Charged, Hessian'
g4mp2_specs = [{
    "method": "ccsd(t)",
    "basis": "6-31G*",
    "tag": "g4mp2_tce",  # Means to use 8 cores per rank
    "keywords": {
        "scf__uhf": True, "tce__freeze": True,
        "ccsd__freeze": "atomic", "qc_module": True,
    }
}, {
    "method": "scf",
    "basis": "G3MP2largeXP",
    "tag": "g4mp2",  # Uses 2 cores per rank
    "keywords": {"scf__uhf": True}
}, {
    "method": "scf",
    "basis": "g4mp2-aug-cc-pvqz",
    "tag": "g4mp2",
    "keywords": {"scf__uhf": True}
}, {
    "method": "scf",
    "basis": "g4mp2-aug-cc-pvtz",
    "tag": "g4mp2",
    "keywords": {"scf__uhf": True}
}, {
    "method": "mp2",
    "basis": "G3MP2largeXP",
    "tag": "g4mp2",
    "keywords": {
        "scf__uhf": True, "tce__freeze": True, "ccsd__freeze": "atomic"
    }
}]
zpe_spec = {
    "method": "slater 0.8 HFexch 0.2 becke88 nonlocal 0.72 vwn_3 0.19 lyp 0.81",
    "basis": "6-31G(2df,p)",
    "tag": "g4mp2_zpe"
}

# Parse input arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--limit', help='Number of molecules to add. -1 to add all', default=1, type=int)
arg_parser.add_argument('password', help='Password for the service')
arg_parser.add_argument('--address', help='Address to QCFractal service', default='localhost:7874', type=str)
arg_parser.add_argument('--file', help='File containing geometries computed from Gaussian', type=str)
args = arg_parser.parse_args()

# Make the FractalClient
client = FractalClient(args.address, verify=False, username='user', password=args.password)

# Make or access the G4MP2 Dataset
#  Energy calculations
colls = client.list_collections(collection_type='Dataset', aslist=True)
if eng_coll_name not in colls:
    print('Initializing dataset')

    # Make the dataset
    eng_coll = Dataset(name=eng_coll_name, client=client)
    eng_coll.set_default_program("nwchem")

else:
    print('Retrieving dataset from server')
    eng_coll = Dataset.from_server(client, name=eng_coll_name)

#  Hessian Calculations
if hess_coll_name not in colls:
    print('Initializing dataset')

    # Make the dataset
    hess_coll = Dataset(name=hess_coll_name, client=client)
    hess_coll.set_default_program("nwchem")
    hess_coll.set_default_driver("hessian")
else:
    print('Retrieving dataset from server')
    hess_coll = Dataset.from_server(client, name=hess_coll_name)


# Load in the molecules to be computed
mols = pd.read_csv(args.file)
print(f'Found {len(mols)}')
if args.limit >= 0:
    mols = mols.iloc[:args.limit]
    print(f'Sampled down to {len(mols)}')

# Add them to the datasets
for coll in [eng_coll, hess_coll]:
    existing_entries = coll.get_entries().index
    for _, mol in tqdm(mols.iterrows()):

        # Do the anion and cation
        for k in ['reduced', 'oxidized']:
            if k == 'oxidized':
                charge = 1
            elif k == 'reduced':
                charge = -1
            else:
                raise ValueError('Look for a typo!')
            molobj = Molecule.from_data(mol[f'xyz_{k}'],
                                        molecular_charge=charge,
                                        name=f'{mol["smiles"]}_{k}_g16')

            if molobj.name not in existing_entries:
                coll.add_entry(molobj.name, molobj)

    coll.save()

# Add in the specifications
#  Energy collection
for spec in g4mp2_specs:
    # Add the keywords to the dataset, use the alias
    spec = spec.copy()
    kw_name = f'{spec["method"]}-{spec["basis"]}'.lower()
    kwset = KeywordSet(values=spec["keywords"],
                       comments=f"Keywords for G4MP2 component: {spec['method']}/{spec['basis']}")
    try:
        eng_coll.get_keywords(alias=kw_name, program='nwchem')
    except KeyError:
        eng_coll.add_keywords(alias=kw_name, program="nwchem", keyword=kwset)
        eng_coll.save()

    spec["keywords"] = kw_name

    # Add the spec to the dataset
    result = eng_coll.compute(program='nwchem', **spec)
eng_coll.save()
print(f'Submitted energy calculations: {result}')

#  Hessian Collection
result = hess_coll.compute(**zpe_spec)
hess_coll.save()
print(f'Submitted hessian calculations: {result}')

