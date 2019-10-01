from edw.actions.initial_geometry import (smiles_to_conformers, optimize_structure,
                                          cluster_and_reduce_conformers)


def test_methane():
    assert len(smiles_to_conformers('C', 8)) >= 1


def test_optimize():
    confs = smiles_to_conformers('C', 1)
    optimize_structure(confs[0])


def test_cluster():
    confs = smiles_to_conformers('CCCCCC', 8)
    assert len(cluster_and_reduce_conformers(confs, 4)) < 8
