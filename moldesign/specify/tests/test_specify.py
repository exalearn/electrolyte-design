from pathlib import Path

from pytest import fixture

from moldesign.specify import FidelityLevel
from moldesign.store.models import MoleculeData, OxidationState


@fixture()
def example_record() -> MoleculeData:
    return MoleculeData.parse_file(Path(__file__).parent / 'example.json')


def test_get_required(example_record):
    # We need to do a relaxation to get the adiabatic EA @ SMB level
    level = FidelityLevel(
        recipe='smb-vacuum-no-zpe',
        model_path='.',
        model_type='mpnn'
    )
    required = level.get_required_calculations(example_record, OxidationState.REDUCED)
    assert len(required) == 1
    assert required[0][-1]  # Means it is a relaxation

    # We need to do a single-point for the vertical IP @ SMB level
    level = FidelityLevel(
        recipe='smb-vacuum-vertical',
        model_path='.',
        model_type='mpnn'
    )
    required = level.get_required_calculations(example_record, OxidationState.OXIDIZED)
    assert len(required) == 1
    assert not required[0][-1]  # Means it is a single-point
    assert required[0][-2] is None  # Means it is in vacuum
    assert required[0][-3] == -1  # Charge of -1
    assert required[0][0] == 'small_basis'
