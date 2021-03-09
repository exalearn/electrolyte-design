from math import isclose

from qcelemental.models import OptimizationResult, AtomicResult

from moldesign.store.models import MoleculeData, OxidationState, IonizationEnergyRecipe


def test_from_smiles():
    md = MoleculeData.from_identifier('C')
    assert md.identifier['smiles'] == 'C'
    assert md.identifier['inchi'] == 'InChI=1S/CH4/h1H4'
    assert md.key == "VNWKTOKETHGBQD-UHFFFAOYSA-N"


def test_from_inchi():
    md = MoleculeData.from_identifier(inchi="InChI=1S/CH4/h1H4")
    assert md.identifier['inchi'] == 'InChI=1S/CH4/h1H4'
    assert md.key == "VNWKTOKETHGBQD-UHFFFAOYSA-N"


def test_add_data():
    md = MoleculeData.from_identifier("O")

    # Load the xtb geometry
    xtb_geom = OptimizationResult.parse_file('records/xtb-neutral.json')
    md.add_geometry(xtb_geom)
    assert "xtb" in md.data
    assert "neutral" in md.data["xtb"]
    assert isclose(md.data["xtb"][OxidationState.NEUTRAL].atomization_energy["xtb-no_zpe"], -0.515, abs_tol=1e-2)
    assert ("xtb", "neutral") == md.match_geometry(xtb_geom.final_molecule.to_string("xyz"))

    # Load in a relaxed oxidized geometry
    xtb_geom = OptimizationResult.parse_file('records/xtb-oxidized.json')
    md.add_geometry(xtb_geom)
    assert "xtb" in md.data
    assert "oxidized" in md.data["xtb"]
    assert ("xtb", "oxidized") == md.match_geometry(xtb_geom.final_molecule.to_string("xyz"))
    assert ("xtb", "neutral") == md.match_geometry(xtb_geom.initial_molecule.to_string("xyz"))

    # Load in a oxidized energy for the neutral structure
    xtb_energy = AtomicResult.parse_file('records/xtb-neutral_xtb-oxidized-energy.json')
    md.add_single_point(xtb_energy)

    # Show that we can compute a redox potential
    recipe = IonizationEnergyRecipe(name="xtb-vertical", geometry_level="xtb", energy_level="xtb", adiabatic=False)
    result = recipe.compute_ionization_potential(md, OxidationState.OXIDIZED)
    assert md.oxidation_potential['xtb-vertical'] == result

    recipe = IonizationEnergyRecipe(name="xtb", geometry_level="xtb", energy_level="xtb", adiabatic=True)
    result = recipe.compute_ionization_potential(md, OxidationState.OXIDIZED)
    assert md.oxidation_potential['xtb'] == result
    assert md.oxidation_potential['xtb'] < md.oxidation_potential['xtb-vertical']

    # Add a single point small_basis computation
    smb_hessian = AtomicResult.parse_file('records/xtb-neutral_smb-neutral-hessian.json')
    md.add_single_point(smb_hessian)
    assert isclose(md.data["xtb"][OxidationState.NEUTRAL].zpe[OxidationState.NEUTRAL]['small_basis'],
                   0.02155, abs_tol=1e-3)
