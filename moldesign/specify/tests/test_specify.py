from pytest import fixture

from moldesign.specify import MultiFidelitySpecification
from moldesign.store.models import MoleculeData, OxidationState


@fixture()
def example_record() -> MoleculeData:
    return MoleculeData.from_identifier(smiles='O')


def test_get_required(example_record):
    levels = MultiFidelitySpecification(levels=['xtb-vacuum', 'xtb-acn'])

    # Nothing has been done, so our first step will be level 1
    assert levels.get_next_step(example_record, OxidationState.OXIDIZED) == 'xtb-vacuum'
    assert levels.get_next_step(example_record, OxidationState.REDUCED) == 'xtb-vacuum'

    # Let's add an IP value @ xtb-vacuum, we then are ready for xtb-acn for that level
    example_record.oxidation_potential['xtb-vacuum'] = 1

    assert levels.get_next_step(example_record, OxidationState.OXIDIZED) == 'xtb-acn'
    assert levels.get_next_step(example_record, OxidationState.REDUCED) == 'xtb-vacuum'

    # Let's add IP value @ xtb-acn, which means we are done!
    example_record.oxidation_potential['xtb-acn'] = 2

    assert levels.get_next_step(example_record, OxidationState.OXIDIZED) is None
    assert levels.get_next_step(example_record, OxidationState.REDUCED) == 'xtb-vacuum'
