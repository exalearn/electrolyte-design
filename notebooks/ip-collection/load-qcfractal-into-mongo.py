"""Load summary of data from QCFractal into MongoDB"""
from tqdm import tqdm

from moldesign.simulate.thermo import compute_frequencies
from moldesign.simulate.qcfractal import GeometryDataset, SolvationEnergyDataset, HessianDataset
from moldesign.store.mongo import MoleculePropertyDB

# Log in to MongoDB
mongo = MoleculePropertyDB.from_connection_info()

# Get the QCFractal datasets
xtb_geom = GeometryDataset('Electrolyte Geometry XTB', 'xtb')
sbn_geom = GeometryDataset('Electrolyte Geometry NWChem', 'small_basis')
nbn_geom = GeometryDataset('Electrolyte Geometry NWChem, 6-31G(2df,p)', 'normal_basis')

sbn_hess = HessianDataset('Electrolyte Hessian', 'nwchem', 'small_basis')
nbn_hess = HessianDataset('Electrolyte Hessian, 6-31G(2df,p)', 'nwchem', 'normal_basis')

xtb_solv = SolvationEnergyDataset('EDW XTB Solvation Energy', 'xtb', 'xtb')
sbn_solv = SolvationEnergyDataset('EDW NWChem Solvation Energy', 'nwchem', 'small_basis')

# Start with the geometries
for geom in [xtb_geom, sbn_geom, nbn_geom]:
    # Load the records
    records = geom.get_complete_records()
    mols = geom.get_geometries()

    # Put them in the database
    for name, record in tqdm(records.items(), desc=f'geom: {geom.qc_spec}'):
        inchi, state = name.split("_")

        # Store the geometry and total energy
        mol = mongo.get_molecule_record(inchi=inchi)
        mol.subsets.append('initial')
        mol.geometries[state][geom.qc_spec] = mols[inchi][state].to_string('xyz')
        mol.total_energies[state][geom.qc_spec] = record.get_final_energy()

        # Update the MongoDB
        mongo.update_molecule(mol)

# Now, do the Hessians
for hess in [nbn_hess, sbn_hess]:
    # Pre-load the geometries
    mols = hess.get_geometries()

    # Add each hessian
    for name, record in tqdm(hess.get_complete_records().items(), desc=f'hess: {hess.qc_spec}'):
        inchi, state = name.split("_")

        # Get the current record for this object
        #  Only retrieve the data needed to compute the atomization energy and ZPE
        mol = mongo.get_molecule_record(inchi=inchi)

        # Compute the vibrational frequencies, atomization and ZPE
        freqs = compute_frequencies(record.return_result, mols[inchi][state])
        if state not in mol.vibrational_modes:
            mol.vibrational_modes[state] = {}
        mol.vibrational_modes[state][hess.qc_spec] = freqs.tolist()
        mol.update_thermochem(verbose=False)
        assert hess.qc_spec in mol.zpes[state]

        mongo.update_molecule(mol)
