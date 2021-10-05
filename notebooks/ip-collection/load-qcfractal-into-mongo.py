"""Load summary of data from QCFractal into MongoDB"""
from tqdm import tqdm

from moldesign.simulate.qcfractal import GeometryDataset, SolvationEnergyDataset, HessianDataset, SinglePointDataset
from moldesign.store.models import UnmatchedGeometry
from moldesign.store.mongo import MoleculePropertyDB

# Log in to MongoDB
mongo = MoleculePropertyDB.from_connection_info()

# Get the QCFractal datasets
relax_datasets = [
    GeometryDataset('Electrolyte Geometry XTB', 'xtb'),
    GeometryDataset('Electrolyte Geometry NWChem', 'small_basis'),
    GeometryDataset('Electrolyte Geometry NWChem, 6-31G(2df,p)', 'normal_basis')
]

single_point_energy_datasets = [
    # Verticals using XTB geometry
    SinglePointDataset('Electrolyte XTB Neutral Geometry, Small-Basis Energy', 'nwchem', 'small_basis', ),
    SinglePointDataset('Electrolyte XTB Neutral Geometry, Normal-Basis Energy', 'nwchem', 'normal_basis'),
    SinglePointDataset('Electrolyte XTB Neutral Geometry, Diffuse-Basis Energy', 'nwchem', 'diffuse_basis'),

    # Verticals using SMB
    SinglePointDataset('Electrolyte SMB Neutral Geometry, Small-Basis Energy', 'nwchem', 'small_basis'),
    SinglePointDataset('Electrolyte SMB Neutral Geometry, Normal-Basis Energy', 'nwchem', 'normal_basis'),

    # Quasi-adiabatic based on SMB
    SinglePointDataset('Electrolyte SMB Adiabatic Geometry, Normal-Basis Energy', 'nwchem', 'normal_basis'),
    SinglePointDataset('Electrolyte SMB Adiabatic Geometry, Diffuse-Basis Energy', 'nwchem', 'diffuse_basis')
]

hessian_datasets = [
    HessianDataset('Electrolyte Hessian', 'nwchem', 'small_basis'),
    HessianDataset('Electrolyte Hessian, 6-31G(2df,p)', 'nwchem', 'normal_basis')
]

solvation_energy_datasets = [
    # Adiabatic
    SolvationEnergyDataset('EDW XTB Solvation Energy', 'xtb', 'xtb'),
    SolvationEnergyDataset('EDW NWChem Solvation Energy', 'nwchem', 'small_basis'),
    SolvationEnergyDataset('EDW Normal-Basis Solvation Energy', 'nwchem', 'normal_basis'),

    # Vertical using XTB geometry
    SolvationEnergyDataset('Electrolyte XTB Neutral Geometry, Small-Basis Solvation Energy', 'nwchem', 'small_basis'),
    SolvationEnergyDataset('Electrolyte XTB Neutral Geometry, Normal-Basis Solvation Energy', 'nwchem', 'normal_basis'),
    SolvationEnergyDataset('Electrolyte XTB Neutral Geometry, Diffuse-Basis Solvation Energy', 'nwchem',
                           'diffuse_basis')
]

# Start with the geometries
for geom in relax_datasets:
    # Load the records
    records = geom.get_complete_records()

    # Put them in the database
    for name, record in tqdm(records.items(), desc=f'geom: {geom.qc_spec}', total=len(records)):
        inchi, state = name.split("_")

        # Store the geometry and total energy
        mol = mongo.get_molecule_record(inchi=inchi)
        mol.subsets.append('initial')
        try:
            mol.add_geometry(record, overwrite=True, client=geom.client)
        except ValueError as e:
            print(f'ERROR: {str(e)}')

        # Update the MongoDB
        mongo.update_molecule(mol)

# Do the vertical energies
for data in single_point_energy_datasets:
    records = data.get_complete_records()

    # Put them in the database
    for name, record in tqdm(records.items(), desc=f'spe: {data.qc_spec}', total=len(records)):
        inchi, state = name.split("_")

        # Store the geometry and total energy
        mol = mongo.get_molecule_record(inchi=inchi)
        mol.add_single_point(record, client=data.client)

        # Update the MongoDB
        mongo.update_molecule(mol)

# Do the solvation energies
for data in solvation_energy_datasets:
    records = data.get_complete_records()

    # Put them in the database
    missed = 0
    for name, record in tqdm(records.items(), desc=f'solv: {data.qc_spec}', total=len(records)):
        inchi, state = name.split("_")

        # Store the geometry and total energy
        mol = mongo.get_molecule_record(inchi=inchi)
        try:
            mol.add_single_point(record, client=data.client)
        except UnmatchedGeometry:
            missed += 1
            continue

        # Update the MongoDB
        mongo.update_molecule(mol)
    if missed > 0:
        print(f'Could not match {missed} computations')

# Now, do the Hessians
for hess in hessian_datasets:
    records = hess.get_complete_records()

    # Add each hessian
    for name, record in tqdm(records, desc=f'hess: {hess.qc_spec}', total=len(records)):
        inchi, state = name.split("_")

        # Get the current record for this object
        #  Only retrieve the data needed to compute the atomization energy and ZPE
        mol = mongo.get_molecule_record(inchi=inchi)
        mol.add_single_point(record, client=hess.client)

        # Update the record, which will compute the ZPE and such
        mongo.update_molecule(mol)
