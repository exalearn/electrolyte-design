from edw.initial_geometry import smiles_to_conformers, optimize_structure


def test_methane():
    assert len(smiles_to_conformers('C', 8)) >= 1


def test_optimize():
    confs = smiles_to_conformers('C', 1)
    optimize_structure(confs[0])
