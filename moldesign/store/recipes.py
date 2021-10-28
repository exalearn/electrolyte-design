"""List of recipes for computing functional properties of molecules"""

from .models import RedoxEnergyRecipe, MoleculeData, OxidationState

redox_recipes = [
    RedoxEnergyRecipe(name='xtb-vacuum-vertical', geometry_level='xtb', energy_level='xtb', adiabatic=False),
    RedoxEnergyRecipe(name='xtb-vacuum', geometry_level='xtb', energy_level='xtb', adiabatic=True),
    RedoxEnergyRecipe(name='xtb-acn', geometry_level='xtb', energy_level='xtb', adiabatic=True,
                      solvent='acetonitrile', solvation_level='xtb'),
    RedoxEnergyRecipe(name='smb-vacuum-vertical', geometry_level='small_basis',
                      energy_level='small_basis', adiabatic=False),
    RedoxEnergyRecipe(name='smb-vacuum-no-zpe', geometry_level='small_basis',
                      energy_level='small_basis', adiabatic=True),
    RedoxEnergyRecipe(name='smb-vacuum', geometry_level='small_basis',
                      energy_level='small_basis', zpe_level='small_basis', adiabatic=True),
    RedoxEnergyRecipe(name='nob-vacuum-smb-geom', geometry_level='small_basis',
                      energy_level='normal_basis', adiabatic=True),
    RedoxEnergyRecipe(name='nob-acn-smb-geom', solvent='acetntrl', geometry_level='small_basis',
                      energy_level='normal_basis', solvation_level='normal_basis', adiabatic=True),
    RedoxEnergyRecipe(name='dfb-vacuum-smb-geom', geometry_level='small_basis',
                      energy_level='diffuse_basis', adiabatic=True),
]


def get_recipe_by_name(name: str) -> RedoxEnergyRecipe:
    """Lookup a specific recipe

    Args:
        name: Name of the recipe
    Returns:
        Desired recipe
    """

    for recipe in redox_recipes:
        if recipe.name == name:
            return recipe
    raise KeyError(f'Recipe not found: {name}')


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
