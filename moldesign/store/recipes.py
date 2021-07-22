"""List of recipes for computing functional properties of molecules"""

from .models import IonizationEnergyRecipe, MoleculeData, OxidationState


redox_recipes = [
    IonizationEnergyRecipe(name='xtb-vacuum', geometry_level='xtb', energy_level='xtb', adiabatic=True),
    IonizationEnergyRecipe(name='smb-vacuum-vertical', geometry_level='small_basis',
                           energy_level='small_basis', adiabatic=False),
    IonizationEnergyRecipe(name='smb-vacuum-no-zpe', geometry_level='small_basis',
                           energy_level='small_basis', adiabatic=True),
    IonizationEnergyRecipe(name='smb-vacuum', geometry_level='small_basis',
                           energy_level='small_basis', zpe_level='small_basis', adiabatic=True),
]


def apply_recipes(data: MoleculeData):
    """Apply all of the redox computation recipes to a molecule
    to update its records

    Args:
        data: Molecule data to be updated
    """

    for state in [OxidationState.OXIDIZED, OxidationState.REDUCED]:
        for recipe in redox_recipes:
            try:
                recipe.compute_ionization_potential(data, state)
            except (ValueError, KeyError):
                continue
