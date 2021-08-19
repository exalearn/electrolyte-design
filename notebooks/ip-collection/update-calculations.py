"""Submit new calculations"""
from typing import List

from qcelemental.models import Molecule

from moldesign.simulate.qcfractal import GeometryDataset, SolvationEnergyDataset, HessianDataset, SinglePointDataset

# List the solvents to use
_xtb_solvs = ['water', 'acetonitrile']
_nwc_solvs = ['acetntrl']

# Get the geometry datasets
xtb_geom = GeometryDataset('Electrolyte Geometry XTB', 'xtb')
smb_geom = GeometryDataset('Electrolyte Geometry NWChem', 'small_basis')
nbn_geom = GeometryDataset('Electrolyte Geometry NWChem, 6-31G(2df,p)', 'normal_basis')

smb_xtb_vert = SinglePointDataset('Electrolyte XTB Neutral Geometry, Small-Basis Energy', 'nwchem', 'small_basis',)
nbn_xtb_vert = SinglePointDataset('Electrolyte XTB Neutral Geometry, Normal-Basis Energy', 'nwchem', 'normal_basis')
dif_xtb_vert = SinglePointDataset('Electrolyte XTB Neutral Geometry, Diffuse-Basis Energy', 'nwchem', 'diffuse_basis')

smb_smb_vert = SinglePointDataset('Electrolyte SMB Neutral Geometry, Small-Basis Energy', 'nwchem', 'small_basis')
nbn_smb_vert = SinglePointDataset('Electrolyte SMB Neutral Geometry, Normal-Basis Energy', 'nwchem', 'normal_basis')

nbn_smb_adia = SinglePointDataset('Electrolyte SMB Adiabatic Geometry, Normal-Basis Energy', 'nwchem', 'normal_basis')
dif_smb_adia = SinglePointDataset('Electrolyte SMB Adiabatic Geometry, Diffuse-Basis Energy', 'nwchem', 'diffuse_basis')

sbn_hess = HessianDataset('Electrolyte Hessian', 'nwchem', 'small_basis')
nbn_hess = HessianDataset('Electrolyte Hessian, 6-31G(2df,p)', 'nwchem', 'normal_basis')

xtb_solv = SolvationEnergyDataset('EDW XTB Solvation Energy', 'xtb', 'xtb', _xtb_solvs)
smb_solv = SolvationEnergyDataset('EDW NWChem Solvation Energy', 'nwchem', 'small_basis', _nwc_solvs)
nbn_solv = SolvationEnergyDataset('EDW Normal-Basis Solvation Energy', 'nwchem', 'normal_basis', _nwc_solvs)

smb_xtb_vert_solv = SolvationEnergyDataset('Electrolyte XTB Neutral Geometry, Small-Basis Solvation Energy', 'nwchem', 'small_basis', _nwc_solvs)
nbn_xtb_vert_solv = SolvationEnergyDataset('Electrolyte XTB Neutral Geometry, Normal-Basis Solvation Energy', 'nwchem', 'normal_basis', _nwc_solvs)
dif_xtb_vert_solv = SolvationEnergyDataset('Electrolyte XTB Neutral Geometry, Diffuse-Basis Solvation Energy', 'nwchem', 'diffuse_basis', _nwc_solvs)

# Start charged geometries if a neutral is completed
for geom in [xtb_geom, smb_geom, nbn_geom]:
    n_started = geom.start_charged_geometries()
    print(f'Started {n_started} charged geometries for {geom.coll.name}')

# Begin Hessian calculations for the NWChem runs
for hess, geom in [(sbn_hess, smb_geom), (nbn_hess, nbn_geom)]:
    was_added = hess.add_geometries(geom)
    print(f'Added {was_added} geometries from {geom.coll.name} to {hess.coll.name}')
    num_started = hess.start_computation()
    print(f'Started {num_started} computations for {hess.coll.name}')


# Start computations with neutral geometries
def submit_vertical_geometries(geom_dataset: GeometryDataset, vert_datasets: List[SinglePointDataset]):
    all_geoms = geom_dataset.get_geometries()
    print(f'Found {len(all_geoms)} molecules in {geom_dataset.coll.name}')
    for inchi, geoms in all_geoms.items():
        # Get the neutral geometry
        if 'neutral' not in geoms:
            continue
        geom = geoms['neutral'].to_string('xyz')

        # Start the neutral geometry in all three charge states
        for postfix, charge in zip(['reduced', 'neutral', 'oxidized'], [-1, 0, 1]):
            # Make a name
            if charge != 0:
                identifier = f'{inchi}_xtb_neutral_{postfix}'
            else:
                identifier = f'{inchi}_xtb_neutral'
            new_geom = Molecule.from_data(geom, 'xyz', molecular_charge=charge, name=identifier)
            # Loop over the different levels of accuracy
            for vert in vert_datasets:
                vert.add_molecule(new_geom, inchi, save=False)

    for vert in vert_datasets:  # Start the computations
        vert.coll.save()
        vert_started = vert.start_computation()
        print(f'Started {vert_started} computations for {vert.coll.name}')


submit_vertical_geometries(xtb_geom, [smb_xtb_vert, nbn_xtb_vert, dif_xtb_vert])
submit_vertical_geometries(smb_geom, [smb_smb_vert, nbn_smb_vert])


# Start computations with adiabatic geometries from lower level of theory
def submit_adiabatic_geoemtries(geom_dataset: GeometryDataset, adia_datasets: List[SinglePointDataset]):
    # Loop over all geometries in the source dataset
    all_geoms = geom_dataset.get_geometries()
    print(f'Found {len(all_geoms)} molecules in {geom_dataset.coll.name}')
    
    # Add the molecules
    for inchi, geoms in all_geoms.items():
        for geom in geoms.values():
            for adia in adia_datasets:
                adia.add_molecule(geom, inchi, save=False)
                
    for adia in adia_datasets:  # Start the computations
        adia.coll.save()
        adia_started = adia.start_computation()
        print(f'Started {adia_started} computations for {adia.coll.name}')
            
submit_adiabatic_geoemtries(smb_geom, [nbn_smb_adia, dif_smb_adia])

# Pass geometries from one level forward to the next
for start_geom, end_geom in [(xtb_geom, smb_geom), (smb_geom, nbn_geom)]:
    n_added = 0
    for inchi, geoms in start_geom.get_geometries().items():
        if 'neutral' in geoms:
            was_added = end_geom.add_molecule_from_geometry(geoms['neutral'], inchi=inchi, save=False)
            if was_added:
                n_added += 1
    end_geom.coll.save()
    print(f'Added {n_added} geometries from {start_geom.coll.name} to {end_geom.coll.name}')
    n_started = end_geom.start_compute()
    print(f'Started {n_started} neutral molecule relaxations for {end_geom.coll.name}')

# Start solvent calculations for adiabatic
#  Loop over combinations of geometry datasets to source geometries and
#   solvation energy datasets to start them
for geom, solv in [(xtb_geom, xtb_solv), (smb_geom, smb_solv), (nbn_geom, nbn_solv)]:
    n_added = solv.add_geometries(geom)
    print(f'Added {n_added} new molecules from {geom.coll.name} to {solv.coll.name}')

    n_started = solv.start_computation()
    print(f'Started {n_started} new computations for {solv.coll.name}')
        
def submit_vertical_geometries(geom_dataset: GeometryDataset, vert_datasets: List[SolvationEnergyDataset]):
    """Submit solvation energy computations for vertical geometries"""
    all_geoms = geom_dataset.get_geometries()
    print(f'Found {len(all_geoms)} molecules in {geom_dataset.coll.name}')
    for inchi, geoms in all_geoms.items():
        # Get the neutral geometry
        if 'neutral' not in geoms:
            continue
        geom = geoms['neutral'].to_string('xyz')

        # Start the neutral geometry in all three charge states
        for postfix, charge in zip(['reduced', 'neutral', 'oxidized'], [-1, 0, 1]):
            # Make a name
            if charge != 0:
                identifier = f'{inchi}_{geom_dataset.qc_spec}_neutral_{postfix}'
            else:
                identifier = f'{inchi}_{geom_dataset.qc_spec}_neutral'
            new_geom = Molecule.from_data(geom, 'xyz', molecular_charge=charge, name=identifier)
            # Loop over the different levels of accuracy
            for vert in vert_datasets:
                vert.add_molecule(new_geom, inchi, save=False)
                
    for vert in vert_datasets:  # Start the computations
        vert.coll.save()
        vert_started = vert.start_computation()
        print(f'Started {vert_started} computations for {vert.coll.name}')

submit_vertical_geometries(xtb_geom, [smb_xtb_vert_solv, nbn_xtb_vert_solv, dif_xtb_vert_solv])
