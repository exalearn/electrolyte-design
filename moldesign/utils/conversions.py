from typing import List

import numpy as np
import networkx as nx
from rdkit import Chem


def convert_rdkit_to_nx(mol: Chem.Mol) -> nx.Graph:
    """Convert a networkx graph to a RDKit Molecule

    Args:
        mol (Chem.RWMol): Molecule to be converted
    Returns:
        (nx.Graph) Graph format of the molecule
    """

    graph = nx.Graph()

    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(),
                       atomic_num=atom.GetAtomicNum(),
                       formal_charge=atom.GetFormalCharge(),
                       chiral_tag=atom.GetChiralTag(),
                       hybridization=atom.GetHybridization(),
                       num_explicit_hs=atom.GetNumExplicitHs(),
                       is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       bond_type=bond.GetBondType())
    return graph


def convert_nx_to_rdkit(graph: nx.Graph) -> Chem.Mol:
    """Convert a networkx graph to a RDKit Molecule

    Args:
        graph (nx.Graph) Graph format of the molecule
    Returns:
        (Chem.RWMol): Molecule to be converted
    """
    mol = Chem.RWMol()

    # Special case: empty-graph
    if graph is None:
        return mol

    atomic_nums = nx.get_node_attributes(graph, 'atomic_num')
    chiral_tags = nx.get_node_attributes(graph, 'chiral_tag')
    formal_charges = nx.get_node_attributes(graph, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(graph, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(graph, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(graph, 'num_explicit_hs')
    node_to_idx = {}
    for node in graph.nodes():
        a = Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(graph, 'bond_type')
    for edge in graph.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    Chem.SanitizeMol(mol)
    return mol


def convert_smiles_to_nx(smiles: str, add_hs: bool = False) -> nx.Graph:
    """Compute a networkx graph from a SMILES string

    Args:
        smiles (str): SMILES string to be parsed
        add_hs (str): Whether to add explicit hydrogens
    Returns:
        (nx.Graph) NetworkX representation of the molecule
    """

    mol = Chem.MolFromSmiles(smiles)
    if add_hs:
        mol = Chem.AddHs(mol)

    return convert_rdkit_to_nx(mol)


def convert_nx_to_smiles(graph: nx.Graph) -> str:
    """Compute a SMILES string from a networkx graph"""
    return Chem.MolToSmiles(convert_nx_to_rdkit(graph))


def convert_nx_to_dict(graph: nx.Graph, atom_types: List[int], bond_types: List[str]) -> dict:
    """Convert networkx representation of a molecule to an MPNN-ready dict

    Args:
        graph: Molecule to be converted
        atom_types: Lookup table of observed atom types
        bond_types: Lookup table of observed bond types
    Returns:
        (dict) Molecule as a dict
    """

    # Get the atom types
    atom_type = [n['atomic_num'] for _, n in graph.nodes(data=True)]
    atom_type_id = list(map(atom_types.index, atom_type))

    # Get the bond types, making the data
    connectivity = []
    edge_type = []
    for a, b, d in graph.edges(data=True):
        connectivity.append([a, b])
        connectivity.append([b, a])
        edge_type.append(str(d['bond_type']))
        edge_type.append(str(d['bond_type']))
    edge_type_id = list(map(bond_types.index, edge_type))

    # Sort connectivity array by the first column
    #  This is needed for the MPNN code to efficiently group messages for
    #  each node when performing the message passing step
    connectivity = np.array(connectivity)
    if connectivity.size > 0:
        # Skip a special case of a molecule w/o bonds
        inds = np.lexsort((connectivity[:, 1], connectivity[:, 0]))
        connectivity = connectivity[inds, :]

        # Tensorflow's "segment_sum" will cause problems if the last atom
        #  is not bonded because it returns an array
        if connectivity.max() != len(atom_type) - 1:
            smiles = convert_nx_to_smiles(graph)
            raise ValueError(f"Problem with unconnected atoms for {smiles}")
    else:
        connectivity = np.zeros((0, 2))

    return {
        'n_atom': len(atom_type),
        'n_bond': len(edge_type),
        'atom': atom_type_id,
        'bond': edge_type_id,
        'connectivity': connectivity
    }


def convert_smiles_to_dict(smiles: str, atom_types: List[int], bond_types: List[str], add_hs: bool = False) -> dict:
    """Convert networkx representation of a molecule to an MPNN-ready dict

    Args:
        smiles: Molecule to be converted
        atom_types: Lookup table of observed atom types
        bond_types: Lookup table of observed bond types
        add_hs: Whether to add explicit hydrogens
    Returns:
        (dict) Molecule as a dict
    """ 
    
    graph = convert_smiles_to_nx(smiles, add_hs=add_hs)
    return convert_nx_to_dict(graph, atom_types, bond_types)
