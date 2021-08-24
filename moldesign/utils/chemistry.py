"""Utility operations to perform common chemistry tasks"""

from rdkit import Chem


def get_baseline_charge(smiles: str) -> int:
    """Determine the charge on a molecule from its SMILES string

    Examples:
        H<sub>2</sub>O has a baseline charge of 0
        NH<sub>4</sub>+ has a baseline charge of +1

    Args:
        smiles: SMILES string of the molecule
    Returns:
        Charge on the molecule
    """

    mol = Chem.MolFromSmiles(smiles)
    return Chem.GetFormalCharge(mol)
