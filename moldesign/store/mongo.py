"""Thin wrapper over MongoDB"""
from typing import List, Tuple, Dict, Any

from pymongo.collection import Collection, UpdateResult
from flatten_dict import flatten

from moldesign.store.models import MoleculeData


def generate_update(moldata: MoleculeData) -> Dict[str, Dict[str, Any]]:
    """Generate an update command for a certain molecule

    Returns:
        A dictionary ready for the "update_one" command
    """

    set_fields = flatten(moldata.dict(exclude_defaults=True, exclude={'key'}), 'dot')
    return {'$set': set_fields}


class MoleculePropertyDB:
    """Wrapper for a MongoDB holding molecular property data"""

    def __init__(self, collection: Collection):
        self.collection = collection

    def initialize_index(self):
        """Prepare a new collection.

        Sets the InChI key field as a key index"""
        return self.collection.create_index('key', unique=True)

    def get_training_set(self, input_fields: List[str], output_fields: List[str]) \
            -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
        """Gather a training set from the existing database.

        Get a set of entries where both the input and output exist.

        Args:
            input_fields: List of fields that must exist
        Returns:
            - Input data: Keys are the requested input fields and values are the values for each matching molecule
            - Output data: Keys are the requested output fields and values are the values for each matching molecule
        """

        # Get the records
        query = dict((v, {'$exists': True}) for v in input_fields + output_fields)
        cursor = self.collection.find(query, input_fields + output_fields)

        # Compile the outputs
        # TODO (wardlt): Consider switching to Numpy arrays for these
        inputs = dict((v, []) for v in input_fields)
        outputs = dict((v, []) for v in output_fields)
        for record in cursor:
            record = flatten(record, 'dot')
            for i in input_fields:
                inputs[i].append(record[i])
            for o in output_fields:
                outputs[o].append(record[o])

        return inputs, outputs

    def update_molecule(self, molecule: MoleculeData) -> UpdateResult:
        """Update the data for a single molecule

        Args:
            molecule: Data for a certain molecule to be updated.
                All fields specified in this record will be updated or added to the matching document.
                No fields will be deleted by this operation.
        Returns:
            An update result
        """
        update_record = generate_update(molecule)
        return self.collection.update_one({'key': molecule.key}, update_record, upsert=True)
