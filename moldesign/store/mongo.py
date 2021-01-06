"""Thin wrapper over MongoDB"""
from typing import List, Tuple, Dict, Any, Set, Optional

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
            output_fields: Which fields to produce in the output fields
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

    def get_eligible_molecules(self, input_fields: List[str], output_fields: List[str]) \
            -> Dict[str, List[Any]]:
        """Gather molecules which are eligible for running a certain computation

        Determines eligibility if certain input_fields are populated and the
        output fields _are not set_.

        Args:
            input_fields: List of fields that must exist
            output_fields: Which fields must not exist
        Returns:
            A dictionary with keys of each input field. This will always include "identifiers.inchi" and "key"
        """

        # Get the "exists" query fields
        must_exist = input_fields + ['identifiers.inchi', 'key']

        # Build the query
        query = dict((v, {'$exists': True}) for v in must_exist)
        for f in output_fields:
            query[f] = {'$exists': False}

        # Run the query
        cursor = self.collection.find(query, must_exist)

        # Gather the inputs
        inputs = dict((v, []) for v in must_exist)
        for record in cursor:
            record = flatten(record, 'dot')
            for i in must_exist:
                inputs[i].append(record[i])
        return inputs

    def update_molecule(self, molecule: MoleculeData) -> UpdateResult:
        """Update the data for a single molecule

        Args:
            molecule: Data for a certain molecule to be updated.
                All fields specified in this record will be updated or added to the matching document.
                No fields will be deleted by this operation.
        Returns:
            An update result
        """
        molecule.update_thermochem()  # Ensure all derived fields are computed, if available
        update_record = generate_update(molecule)
        return self.collection.update_one({'key': molecule.key}, update_record, upsert=True)

    def get_molecules(self, match: Optional[Dict] = None, output_key: str = 'identifiers.inchi') -> Set[str]:
        """Get all of the molecules in that match a certain query

        Returns a query of all of their object"""

        return set(self.collection.distinct(output_key, filter=match))
