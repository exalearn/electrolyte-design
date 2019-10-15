"""Functions related to generating initial geometries for quantum chemistry codes"""

import numpy as np
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import AgglomerativeClustering


def compute_inchi_key(smiles: str) -> str:
    """Generate the InChI key for a molecule

    This will be used as an ID for the molecule in eventual databases

    Args:
        smiles (str): SMILES string for a molecule
    Returns:
         (str): InChI key
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol)


def smiles_to_conformers(smiles: str, n_conformers: int,
                         relax: bool = True) -> List[str]:
    """Generate a series of conformers for a molecule

    Args:
        smiles (str): SMILES string for molecule of interest
        n_conformers (int): Number of conformers to generate
        relax (bool): Whether to
    Returns:
        ([str]): List of conformers in Mol format
    """

    # Make an RDK model of the molecules
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)

    # Generate conformers, with a pruning if RMS is less than 0.1 Angstrom
    ids = AllChem.EmbedMultipleConfs(m, numConfs=n_conformers,
                                     pruneRmsThresh=1)

    # If desired, relax the conformers
    if relax:
        AllChem.MMFFOptimizeMoleculeConfs(m)

    # Print out the conformers in Mol format
    return [Chem.MolToMolBlock(m, confId=i) for i in ids]


def optimize_structure(mol: str) -> str:
    """Optimize the coordinates of a molecule using MMF94 forcefield

    Args:
        mol (str): String of molecule structure in Mol format
    Returns:
        (str): String of the relaxed structure in mol format
    """

    m = Chem.MolFromMolBlock(mol, removeHs=False)
    while AllChem.MMFFOptimizeMolecule(m) == 1:
        continue
    return Chem.MolToMolBlock(m)


def cluster_and_reduce_conformers(confs: List[str], max_cluster_dist=2) -> List[str]:
    """Cluster the list of conformers and pick only one representative per cluster

    Works by computing the RMS distance between each conformer and using that
    distance as the tool for clustering

    Args:
        confs ([str]): List of conformers
        max_cluster_dist (float): Distance threshold of
    Returns:
        ([str]): Reduced list of conformers
    """

    # Parse all conformer structures
    mols = [AllChem.MolFromMolBlock(m) for m in confs]

    # Compute the RMS distances
    dists = np.zeros((len(confs),)*2)
    for i in range(1, len(dists)):
        for j in range(i):
            rms = AllChem.AlignMol(mols[i], mols[j])
            dists[i, j] = dists[j, i] = rms

    # Get the minimum distance
    min_rms = dists[np.triu_indices(len(confs), 1)].min()
    dist_threshold = min_rms * max_cluster_dist

    # Perform the clustering
    clust = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                    linkage='average', compute_full_tree=True,
                                    distance_threshold=dist_threshold)
    clust_ids = clust.fit_predict(dists)

    # Get the unique indices
    _, uniq_inds = np.unique(clust_ids, return_index=True)

    return [confs[i] for i in uniq_inds]
