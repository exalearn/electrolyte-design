"""Load summary of data from QCFractal into MongoDB"""
from tqdm import tqdm

from moldesign.simulate.thermo import compute_frequencies
from moldesign.simulate.qcfractal import GeometryDataset, SolvationEnergyDataset, HessianDataset, SinglePointDataset
from moldesign.store.mongo import MoleculePropertyDB

# Log in to MongoDB
mongo = MoleculePropertyDB.from_connection_info()

# Get the QCFractal datasets
xtb_geom = GeometryDataset('Electrolyte Geometry XTB', 'xtb')
sbn_geom = GeometryDataset('Electrolyte Geometry NWChem', 'small_basis')
nbn_geom = GeometryDataset('Electrolyte Geometry NWChem, 6-31G(2df,p)', 'normal_basis')

sbn_hess = HessianDataset('Electrolyte Hessian', 'nwchem', 'small_basis')
nbn_hess = HessianDataset('Electrolyte Hessian, 6-31G(2df,p)', 'nwchem', 'normal_basis')

smb_xtb_vert = SinglePointDataset('Electrolyte XTB Neutral Geometry, Small-Basis Energy', 'nwchem', 'small_basis',)
nbn_xtb_vert = SinglePointDataset('Electrolyte XTB Neutral Geometry, Normal-Basis Energy', 'nwchem', 'normal_basis')

xtb_solv = SolvationEnergyDataset('EDW XTB Solvation Energy', 'xtb', 'xtb')
sbn_solv = SolvationEnergyDataset('EDW NWChem Solvation Energy', 'nwchem', 'small_basis')

# Start with the geometries
for geom in [xtb_geom, sbn_geom, nbn_geom]:
    # Load the records
    records = geom.get_complete_records()

    # Put them in the database
    for name, record in tqdm(records.items(), desc=f'geom: {geom.qc_spec}', total=len(records)):
        inchi, state = name.split("_")

        # Store the geometry and total energy
        mol = mongo.get_molecule_record(inchi=inchi)
        mol.subsets.append('initial')
        mol.add_geometry(record, overwrite=True)

        # Update the MongoDB
        mongo.update_molecule(mol)


# Do the vertical energies
for vert in [smb_xtb_vert, nbn_xtb_vert]:
    pass

# Now, do the Hessians
for hess in [nbn_hess, sbn_hess]:
    records = hess.get_complete_records()

    # Add each hessian
    for name, record in tqdm(records, desc=f'hess: {hess.qc_spec}', total=len(records)):
        inchi, state = name.split("_")

        # Get the current record for this object
        #  Only retrieve the data needed to compute the atomization energy and ZPE
        mol = mongo.get_molecule_record(inchi=inchi)
        mol.add_single_point(record)

        # Update the record, which will compute the ZPE and such
        mongo.update_molecule(mol)
