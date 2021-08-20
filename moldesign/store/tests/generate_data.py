"""Generate the data used by the tests"""

from moldesign.simulate.functions import generate_inchi_and_xyz, relax_structure, run_single_point
from moldesign.simulate.specs import get_qcinput_specification


if __name__ == "__main__":
    # Make a water molecule
    inchi, xyz = generate_inchi_and_xyz('O')

    # Generate the neutral geometry with XTB
    xtb_spec, xtb = get_qcinput_specification("xtb")
    xtb_neutral_xyz, _, record = relax_structure(xyz, xtb_spec, code=xtb)
    with open('records/xtb-neutral.json', 'w') as fp:
        print(record.json(), file=fp)

    # Compute the vertical oxidation potential
    record = run_single_point(xtb_neutral_xyz, "energy", xtb_spec, code=xtb, charge=1)
    with open('records/xtb-neutral_xtb-oxidized-energy.json', 'w') as fp:
        print(record.json(), file=fp)

    # Compute the adiabatic oxidation potential
    xtb_oxidized_xyz, _, record = relax_structure(xtb_neutral_xyz, xtb_spec, code=xtb, charge=1)
    with open('records/xtb-oxidized.json', 'w') as fp:
        print(record.json(), file=fp)

    # Compute the solvation energy for the neutral and the oxidized
    xtb_spec, xtb = get_qcinput_specification('xtb', 'acetonitrile')
    record = run_single_point(xtb_neutral_xyz, 'energy', xtb_spec, code=xtb, charge=0)
    with open('records/xtb-neutral_acn.json', 'w') as fp:
        print(record.json(), file=fp)
    record = run_single_point(xtb_oxidized_xyz, 'energy', xtb_spec, code=xtb, charge=1)
    with open('records/xtb-oxidized_acn.json', 'w') as fp:
        print(record.json(), file=fp)

    # Run a single point Hessian with NWChem
    smb_spec, nwchem = get_qcinput_specification("small_basis")
    record = run_single_point(xtb_neutral_xyz, "hessian", smb_spec, code=nwchem)
    with open('records/xtb-neutral_smb-neutral-hessian.json', 'w') as fp:
        print(record.json(), file=fp)
