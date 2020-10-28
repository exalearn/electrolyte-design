"""Submit new calculations"""

from moldesign.simulate.qcfractal import GeometryDataset, SolvationEnergyDataset, HessianDataset

# List the solvents to use
_xtb_solvs = ['water', 'dmso', 'acetone', 'acetonitrile']
_nwc_solvs = ['water', 'dmse', 'acetone', 'ethanol', 'acetntrl']

# Get the geometry datasets
xtb_geom = GeometryDataset('Electrolyte Geometry XTB', 'xtb')
sbn_geom = GeometryDataset('Electrolyte Geometry NWChem', 'small_basis')
nbn_geom = GeometryDataset('Electrolyte Geometry NWChem, 6-31G(2df,p)', 'normal_basis')

sbn_hess = HessianDataset('Electrolyte Hessian', 'nwchem', 'small_basis')
nbn_hess = HessianDataset('Electrolyte Hessian, 6-31G(2df,p)', 'nwchem', 'normal_basis')

xtb_solv = SolvationEnergyDataset('EDW XTB Solvation Energy', 'xtb', 'xtb')
sbn_solv = SolvationEnergyDataset('EDW NWChem Solvation Energy', 'nwchem', 'small_basis')

# Start charged geometries if a previous level of theory has completed
for geom in [xtb_geom, sbn_geom, nbn_geom]:
    n_started = geom.start_charged_geometries()
    print(f'Started {n_started} charged geometries for {geom.coll.name}')

# Begin Hessian calculations for the NWChem runs
for hess, geom in [(sbn_hess, sbn_geom), (nbn_hess, nbn_geom)]:
    was_added = hess.add_geometries(geom)
    print(f'Added {was_added} geometries from {geom.coll.name} to {hess.coll.name}')
    num_started = hess.start_computation()
    print(f'Started {num_started} computations for {hess.coll.name}')

# Pass geometries from one level forward to the next
for start_geom, end_geom in [(xtb_geom, sbn_geom), (sbn_geom, nbn_geom)]:
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
exit()

# Start solvent calculations
n_added = xtb_solv.add_geometries(xtb_geom)
print(f'Added {n_added} new molecules to XTB solvent computer')

n_started = xtb_solv.start_computation(_xtb_solvs)
print(f'Started {n_started} new solvent XTB computations')

n_added = sbn_solv.add_geometries(sbn_geom)
print(f'Added {n_added} new molecules to NWChem solvent computer')

n_started = sbn_solv.start_computation(_nwc_solvs)
print(f'Started {n_started} new solvent NWChem computations')
