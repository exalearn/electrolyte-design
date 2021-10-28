"""List of recipes for computing functional properties of molecules"""

from .models import IonizationEnergyRecipe, MoleculeData, OxidationState

redox_recipes = [
    IonizationEnergyRecipe(name='xtb-vacuum-vertical', geometry_level='xtb', energy_level='xtb', adiabatic=False),
    IonizationEnergyRecipe(name='xtb-vacuum', geometry_level='xtb', energy_level='xtb', adiabatic=True),
    IonizationEnergyRecipe(name='xtb-acn', geometry_level='xtb', energy_level='xtb', adiabatic=True,
                           solvent='acetonitrile', solvation_level='xtb'),
    IonizationEnergyRecipe(name='smb-vacuum-vertical', geometry_level='small_basis',
                           energy_level='small_basis', adiabatic=False),
    IonizationEnergyRecipe(name='smb-vacuum-no-zpe', geometry_level='small_basis',
                           energy_level='small_basis', adiabatic=True),
    IonizationEnergyRecipe(name='smb-vacuum', geometry_level='small_basis',
                           energy_level='small_basis', zpe_level='small_basis', adiabatic=True),
    IonizationEnergyRecipe(name='nob-vacuum-smb-geom', geometry_level='small_basis',
                           energy_level='normal_basis', adiabatic=True),
    IonizationEnergyRecipe(name='nob-acn-smb-geom', solvent='acetntrl', geometry_level='small_basis',
                           energy_level='normal_basis', solvation_level='normal_basis', adiabatic=True),
    IonizationEnergyRecipe(name='dfb-vacuum-smb-geom', geometry_level='small_basis',
                           energy_level='diffuse_basis', adiabatic=True),
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
                recipe.compute_redox_potential(data, state)
            except (ValueError, KeyError):
                continue
