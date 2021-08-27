"""Routines for generating reasonable initial geometries"""

import logging

from rdkit import Chem
from ase.io.xyz import simple_read_xyz, write_xyz
from io import StringIO
import networkx as nx
import numpy as np

from moldesign.utils.conversions import convert_smiles_to_nx

logger = logging.getLogger(__name__)


def fix_cyclopropenyl(xyz: str, smiles: str) -> str:
    """Detect cyclopropenyl groups and assure they are planar.

    Args:
        xyz: Current structure in XYZ format
        smiles: SMILES string of the molecule
    Returns:
        Version of atoms with the rings flattened
    """

    # Find cyclopropenyl groups
    mol = Chem.MolFromSmiles(smiles)
    rings = mol.GetSubstructMatches(Chem.MolFromSmarts("c1c[c+]1"))
    if len(rings) == 0:
        return xyz  # no changes

    # For each ring, flatten it
    atoms = next(simple_read_xyz(StringIO(xyz), slice(None)))
    g = convert_smiles_to_nx(smiles, add_hs=True)
    for ring in rings:
        # Get the normal of the ring
        normal = np.cross(*np.subtract(atoms.positions[ring[:2], :], atoms.positions[ring[2], :]))
        normal /= np.linalg.norm(normal)

        # Adjust the groups attached to each member of the ring
        for ring_atom in ring:
            # Get the ID of the group bonded to it
            bonded_atom = next(r for r in g[ring_atom] if r not in ring)

            # Determine the atoms that are part of that functional group
            h = g.copy()
            h.remove_edge(ring_atom, bonded_atom)
            a, b = nx.connected_components(h)
            mask = np.zeros((len(atoms),), dtype=bool)
            if bonded_atom in a:
                mask[list(a)] = True
            else:
                mask[list(b)] = True

            # Get the rotation angle
            bond_vector = atoms.positions[bonded_atom, :] - atoms.positions[ring_atom, :]
            angle = np.dot(bond_vector, normal) / np.linalg.norm(bond_vector)
            rot_angle = np.arccos(angle) - np.pi / 2
            logger.debug(f'Rotating by {rot_angle} radians')

            # Perform the rotation
            rotation_axis = np.cross(bond_vector, normal)
            atoms._masked_rotate(atoms.positions[ring_atom], rotation_axis, rot_angle, mask)

            # make the atom at a 150 angle with the the ring too
            another_ring = next(r for r in ring if r != ring_atom)
            atoms.set_angle(another_ring, ring_atom, bonded_atom, 150, mask=mask)
            assert np.isclose(atoms.get_angle(another_ring, ring_atom, bonded_atom), 150).all()

            # Make sure it worked
            bond_vector = atoms.positions[bonded_atom, :] - atoms.positions[ring_atom, :]
            angle = np.dot(bond_vector, normal) / np.linalg.norm(bond_vector)
            final_angle = np.arccos(angle)
            assert np.isclose(final_angle, np.pi / 2).all()

        logger.info(f'Detected {len(rings)} cyclopropenyl rings. Ensured they are planar.')

        # Write to a string
        out = StringIO()
        write_xyz(out, [atoms])
        return out.getvalue()
