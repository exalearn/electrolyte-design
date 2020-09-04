"""Submit new calculations"""

from edw.qc import GeometryDataset, SolvationEnergyDataset

# List the solvents to use
_xtb_solvs = ['water', 'dmso', 'acetone', 'acetonitrile']

# Get the geometry datasets
xtb_geom = GeometryDataset('Electrolyte Geometry XTB', 'xtb')
nwc_geom = GeometryDataset('Electrolyte Geometry NWChem', 'small_basis')
xtb_solv = SolvationEnergyDataset('EDW XTB Solvation Energy', 'xtb', 'xtb')

# Start charged geometries for XTB and NWC
xtb_geom.start_charged_geometries()
n_started = xtb_geom.start_compute()
print(f'Started {n_started} charged geometries for XTB')

nwc_geom.start_charged_geometries()
n_started = nwc_geom.start_compute()
print(f'Started {n_started} charged geometries for NWChem')

# Pass neutral XTB geometries forward as starting geometries for NWChem
n_added = 0
for inchi, geoms in xtb_geom.get_geometries().items():
    if 'neutral' in geoms:
        was_added = nwc_geom.add_molecule_from_geometry(geoms['neutral'], inchi=inchi, save=False)
        if was_added:
            n_added += 1
nwc_geom.coll.save()
n_started = nwc_geom.start_compute()
print(f'Added {n_added} geometries to NWChem, started {n_started} neutral molecule relaxations')

# Start solvent calculations
n_added = xtb_solv.add_geometries(xtb_geom)
print(f'Added {n_added} new molecules to solvent computer')

n_started = xtb_solv.start_computation(_xtb_solvs)
print(f'Started {n_started} new solvent computations')
