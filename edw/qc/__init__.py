"""Functions and modules related to high-throughput computational chemistry"""
import os
import logging
from typing import Optional, Dict, Type, ClassVar, List

import numpy as np
import pandas as pd
from qcelemental.models import Molecule

from rdkit import Chem
from qcfractal.interface import FractalClient
from qcelemental.physical_constants import constants
from qcfractal.interface.collections import OptimizationDataset, Dataset
from qcfractal.interface.collections.collection import Collection
from qcfractal.interface.models import ComputeResponse

from edw.qc.specs import get_optimization_specification, create_computation_spec
from edw.qc.thermo import compute_zpe
from edw.utils import generate_inchi_and_xyz

logger = logging.getLogger(__name__)

f = constants.ureg.Quantity('96485.3329 A*s/mol')
e = constants.ureg.Quantity('1.602176634e-19 A*s')
mol = constants.ureg.Quantity('1 mol')


class QCFractalWrapper:
    """Wrapper over a QFractal Dataset class

    Handles creating and authenticating with the underlying class method.

    It is a base class for building superclasses that create utility operations
    for adding molecules to the database (e.g., generate XYZ coordinates to allow
    for a 'just add this SMILES'), using a consistent identifier format,
    post-processing routines for accessing data in a database,
    and passing data between different steps of the process."""

    def __init__(self, coll_name: str, qc_spec: str, base_class: Type[Collection] = Dataset,
                 address: str = "localhost:7874", qca_passwd: Optional[str] = os.environ.get("QCAUSR", None),
                 create: bool = False):
        """Open the geometry computation dataset

        Args:
            address: Address for the QCFractal server
            base_class: Type of the collection
            qc_spec: Name of the QC specification
            coll_name: Name of the collection holding the data
            qca_passwd: Password for the QCFractal server
            create: Whether creating a new collection is acceptable
        """

        self.qc_spec = qc_spec
        self.client = FractalClient(address, username='user', password=qca_passwd, verify=False)
        try:
            self.coll = base_class.from_server(name=coll_name, client=self.client)
        except KeyError as ex:
            if create:
                self.coll = base_class(name=coll_name, client=self.client)
                self.coll.save()
            else:
                raise ex


class GeometryDataset(QCFractalWrapper):
    """Manager for geometry relaxations.

    Determines the neutral geometry starting from an geometry generated from a SMILES string,
    and then the oxidized and reduced geometries.

    Wraps the OptimizationDataset of QCFractal, which allows for running molecular optimizations with
    different settings all from the same starting geometry."""

    coll: ClassVar[OptimizationDataset]

    def __init__(self, coll_name: str, qc_spec: str, **kwargs):
        """
        Args:
            coll_name: Name for the collection
            qc_spec: Name of the QC specification
            kwargs: Passed to the base collection type
        """
        super().__init__(coll_name, qc_spec, base_class=OptimizationDataset, **kwargs)
        try:
            self.add_specification(self.qc_spec)
        except KeyError:
            pass

    def add_molecule_from_smiles(self, smiles: str, save: bool = True, **attributes) -> bool:
        """Add a molecule to the database

        Will generate 3D coordinates with RDKit, assign yet-undetermined stereochemistry based
        on the 3D geometry, generate an InCHi string for the molecules, then add it to the database.

        Does not start any new computations, just adds it as a potential molecule.

        Args:
            smiles: SMILES string of molecule to be added
            save: Whether to save dataset after adding molecule. Use ``False`` if you will be
                submitting many molecules at the same time
            attributes: Any additional information to provide about the molecule
        Returns:
             Whether this molecule was added to the database
        """

        # Generate 3D coordinates and identifier for the molecule
        inchi, xyz = generate_inchi_and_xyz(smiles)
        identifier = f"{inchi}_neutral"

        # Check if the molecule is already in the database
        existing_mols = self.coll.df
        if identifier in existing_mols:
            return False

        # If not, add the molecule
        mol = Molecule.from_data(xyz, name=identifier)
        try:
            self.coll.add_entry(identifier, initial_molecule=mol, save=save, attributes=attributes)
        except KeyError:
            return False
        return True

    def add_molecule_from_geometry(self, mol: Molecule, inchi: str, save: bool = False):
        """Add a molecule to the dataset

        Args:
            mol: Molecule to be added
            save: Whether to save the database after adding the molecule
            inchi: InChI string describing the molecule
        Returns:
            Whether the molecule was added to the dataset
        """

        # Generate an identifier for the molecule
        _charge_state_str = {0: 'neutral', 1: 'oxidized', -1: 'reduced'}
        mol_charge = int(mol.molecular_charge)
        if mol_charge not in _charge_state_str:
            raise ValueError(f'No name for a molecule with a charge of {mol_charge}')
        identifier = f'{inchi}_{_charge_state_str[mol_charge]}'

        # Check if the molecule is already in the database
        existing_mols = self.coll.df
        if identifier in existing_mols:
            return False

        # If not, add the molecule
        try:
            self.coll.add_entry(identifier, initial_molecule=mol, save=save)
        except KeyError:
            return False
        return True

    def add_specification(self, name: str):
        """Adds a new specification for relaxation

        Args:
            name (str): Name of the predefined relaxation specification
        """

        spec = get_optimization_specification(self.client, name)
        self.coll.add_specification(**spec)

    def start_compute(self, tag: Optional[str] = None) -> int:
        """Send all remaining calculations to queue

        Args:
            tag: Tag to use. By default, uses 'edw_{tag}'
        Returns:
            Number of new calculations started
        """
        if tag is None:
            tag = f'edw_{self.qc_spec}'
        return self.coll.compute(self.qc_spec, tag=tag)

    def start_charged_geometries(self, tag: Optional[str] = None) -> int:
        """Start charged geometry calculations for charged geometries

        Searches for molecules which have completed for a certain specification and sends in
        calculations to determine charged geometries.

        Args:
             tag: Tag to use for the molecules being computed
        Returns:
            (int): Number of new molecules started
        """

        # Query for the latest results for this specification
        records = self.coll.query(self.qc_spec, force=True)

        # Get the neutrals which have finished
        mask = records.index.str.endswith('_neutral') & records.map(lambda x: x.status.value == "COMPLETE")
        completed_neutrals = records[mask]
        logger.info(f'Found {len(completed_neutrals)} neutral molecules that have finished')

        # Determine which of these completed neutrals already
        inchis = completed_neutrals.index.map(lambda x: x.split("_")[0])
        mask = [f'{i}_oxidized' not in records.index for i in inchis]
        compute_needed = completed_neutrals[mask]
        logger.info(f'Of these, {len(compute_needed)} neutral molecules have not yet ionized')

        # Insert the ionized molecules
        for name, record in compute_needed.items():
            inchi, _ = name.split("_", 1)
            neutral_mol = record.get_final_molecule().to_string('xyz')
            for charge, postfix in zip([-1, 1], ["reduced", "oxidized"]):
                identifier = f"{inchi}_{postfix}"
                new_mol = Molecule.from_data(neutral_mol, 'xyz', molecular_charge=charge, name=identifier)
                self.coll.add_entry(identifier, new_mol, save=False)
        self.coll.save()  # Update them all at once
        logger.info('Saved all of the new molecules to the database')

        # Start the new calculations!
        return self.start_compute(tag)

    def get_geometries(self) -> Dict[str, Dict[str, Molecule]]:
        """Get all completed geometries

        Returns:
            The geometries in different charge states for each molecule
        """

        # Get the records
        records = self.get_complete_records()

        # Get the molecules
        mol_ids = records.map(lambda x: x.final_molecule).tolist()
        mols: List[Molecule] = []
        for i in range(0, len(mol_ids), 1000):  # Query by 1000s
            mols.extend(self.client.query_molecules(mol_ids[i:i+1000]))
        mol_lookup = dict((m.id, m) for m in mols)

        # Get all of the geometries
        output = {}
        for name, record in records.items():
            inchi, state = name.split("_")
            if inchi not in output:
                output[inchi] = {}
            mol = mol_lookup[record.final_molecule]
            assert record.final_molecule == mol.id
            output[inchi][state] = mol
        return output

    def get_complete_records(self) -> pd.Series:
        """Get all complete geometries

        Returns:
            All of the complete records
        """

        # Query for the latest results for this specification
        records = self.coll.query(self.qc_spec, force=True)
        logger.info(f'Pulled {len(records)} records for {self.qc_spec}')

        # Get only those which have completed
        records = records[records.map(lambda x: x.status.value == "COMPLETE")]
        logger.info(f'Found {len(records)} completed calculations')
        return records

    def get_energies(self) -> Dict[str, Dict[str, float]]:
        """Get the energy for each molecule

        Returns:
            The InChI strings and energy of the structure in different charge states
        """

        # Get the records
        records = self.get_complete_records()

        # Get all of the entries
        output = {}
        for name, record in records.items():
            inchi, state = name.split("_")
            if inchi not in output:
                output[inchi] = {}
            output[inchi][state] = record.get_final_energy()
        return output


class SinglePointDataset(QCFractalWrapper):
    """Base dataset for calculations on a single molecular geometry"""

    coll: ClassVar[Dataset]

    def __init__(self, coll_name: str, code: str, qc_spec: str, **kwargs):
        """
        Args:
            coll_name: Collection name.
            code: Which code to use
            qc_spec: Name of the specification
            **kwargs
        """
        super().__init__(coll_name, qc_spec, base_class=Dataset, **kwargs)
        self.coll.set_default_program(code)
        self.coll.save()

    def add_molecule(self, mol: Molecule, inchi: str, save: bool = True, **attributes) -> bool:
        """Add a molecule to the dataset

        Args:
            mol: Molecule to be added
            save: Whether to save the database after adding the molecule
            inchi: InChI string describing the molecule
        Returns:
            Whether the molecule was added to the dataset
        """

        # Generate an identifier for the molecule
        _charge_state_str = {0: 'neutral', 1: 'oxidized', -1: 'reduced'}
        mol_charge = int(mol.molecular_charge)
        if mol_charge not in _charge_state_str:
            raise ValueError(f'No name for a molecule with a charge of {mol_charge}')
        identifier = f'{inchi}_{_charge_state_str[mol_charge]}'

        # Check if the molecule is already in the database
        existing_mols = self.coll.df
        if identifier in existing_mols:
            return False

        # If not, add the molecule
        try:
            self.coll.add_entry(identifier, molecule=mol, **attributes)
            if save:
                self.coll.save()
        except KeyError:
            return False
        return True

    def add_geometries(self, dataset: GeometryDataset) -> int:
        """Add all geometries from a GeometryDataset

        Args:
            dataset: Dataset from which to pull geometries
        """

        # Query to get the latest molecules
        self.coll.get_values()

        n_added = 0
        assert self.qc_spec == dataset.qc_spec, "Datasets have different quantum chemistry specifications"

        # Get all of the geometries
        for inchi, geoms in dataset.get_geometries().items():
            for geom in geoms.values():
                was_added = self.add_molecule(geom, inchi, save=False)
                if was_added:
                    n_added += 1
        self.coll.save()
        return n_added

    def get_geometries(self, records: Optional[pd.Series] = None)\
            -> Dict[str, Dict[str, Molecule]]:
        """Get all completed geometries

        Args:
            records: Series of records for which to retrieve molecules.
                If not provided, will get all complete records
        Returns:
            The geometries in different charge states for each molecule
        """

        # Get the records
        if records is None:
            records = self.get_complete_records()

        # Get the molecules
        mol_ids = records.map(lambda x: x.molecule).tolist()
        mols: List[Molecule] = []
        for i in range(0, len(mol_ids), 1000):  # Query by 1000s
            mols.extend(self.client.query_molecules(mol_ids[i:i+1000]))
        mol_lookup = dict((m.id, m) for m in mols)

        # Get all of the geometries
        output = {}
        for name, record in records.items():
            inchi, state = name.split("_")
            if inchi not in output:
                output[inchi] = {}
            mol = mol_lookup[record.molecule]
            assert record.molecule == mol.id
            output[inchi][state] = mol
        return output

    def get_complete_records(self) -> pd.Series:
        """Get all complete geometries

        Returns:
            All of the complete records
        """

        # Get the specification
        methods = self.coll.list_records()
        assert len(methods) == 1, 'We should have exactly one method per dataset'
        method = methods.iloc[0]['method']

        # Get the records
        records = self.coll.get_records(method=method).iloc[:, 0]
        logger.info(f'Pulled {len(records)} records for {self.coll.name}')

        # Get only those which have completed
        records = records[~records.isnull()]
        logger.info(f'Found {len(records)} completed calculations')
        return records


class SolvationEnergyDataset(SinglePointDataset):
    """Perform the solvation energy calculations

    Each instance of this dataset should only be used to store geometries from
    a single type of QC method. Keeping the type of calculation consistent simplifies
    the identifiers to needing only the molecular identifier and the charge state.
    The provenance of the geometry is, effectively, encoded in the collection name.
    """

    def __init__(self, coll_name: str, code: str, qc_spec: str, **kwargs):
        """
        Args:
            coll_name: Collection name.
            code: Which code to use
            qc_spec: Name of the specification
            **kwargs
        """
        super().__init__(coll_name, code, qc_spec, **kwargs)
        self.coll.set_default_units('hartree')
        self.coll.save()

    def start_computation(self, solvents: List[str], tag: Optional[str] = None) -> int:
        """Submit calculations for every molecule in a certain list of solvents

        Args:
            solvents: Names of the solvents
            tag: Tag to use for the computations
        Returns:
            Number of calculations started
        """

        # Create a tag, if need be
        if tag is None:
            tag = f'edw_{self.qc_spec}'

        # Query to get the latest molecules
        self.coll.get_values()

        # Start computations for each solvent
        n_submitted = 0
        for solvent in solvents:
            # Generate the specification
            spec = create_computation_spec(self.qc_spec, solvent)

            # Deal with the keywords
            alias = f'{self.qc_spec}_{solvent}'
            try:
                self.coll.get_keywords(alias=alias, program=spec["program"])
            except KeyError:
                self.coll.add_keywords(alias=alias, program=spec["program"], keyword=spec["keywords"])
                self.coll.save()
                logger.info(f'Added keywords {alias} to the collection')

            # Start the computation
            results: ComputeResponse = self.coll.compute(
                method=spec["method"], basis=spec.get("basis", None), keywords=alias, tag=tag
            )
            n_submitted += len(results.submitted)

        return n_submitted

    def get_energies(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get the energies in all solvents"""

        output = {}
        for calc_label, data in self.coll.get_values().items():
            # Get the name of the solvent
            _, solvent = calc_label.rsplit("_", 1)

            # Store all of the solvent computations
            for label, energy in data.items():
                if np.isnan(energy):
                    continue

                # Get the molecule name
                inchi, charge_state = label.rsplit("_", 1)

                # Prepare place for the data
                if inchi not in output:
                    output[inchi] = {}
                if charge_state not in output[inchi]:
                    output[inchi][charge_state] = {}
                output[inchi][charge_state][solvent] = energy
        return output


class HessianDataset(SinglePointDataset):
    """Compute Hessians for a certain dataset"""

    def __init__(self, coll_name: str, code: str, qc_spec: str, **kwargs):
        """
        Args:
            coll_name: Collection name.
            code: Which code to use
            qc_spec: Name of the specification
            **kwargs
        """
        super().__init__(coll_name, code, qc_spec, **kwargs)
        self.coll.set_default_units('hartree / angstrom ** 2')
        self.coll.set_default_driver('hessian')
        self.coll.save()

    def start_computation(self, tag: Optional[str] = None) -> int:
        """Begin Hessian computations for a certain list of solvents

        Args:
            tag: Tag to use for the computations
        Returns:
            Number of calculations started
        """

        # Create a tag, if need be
        if tag is None:
            tag = f'edw_{self.qc_spec}'

        # Query to get the latest molecules
        self.coll.get_values()

        # Start computations for each solvent
        spec = create_computation_spec(self.qc_spec)

        # Deal with the keywords
        alias = f'{self.qc_spec}'
        try:
            self.coll.get_keywords(alias=alias, program=spec["program"])
        except KeyError:
            self.coll.add_keywords(alias=alias, program=spec["program"], keyword=spec["keywords"])
            self.coll.save()
            logger.info(f'Added keywords {alias} to the collection')

        # Start the computation
        results: ComputeResponse = self.coll.compute(
            method=spec["method"], basis=spec.get("basis", None), keywords=alias, tag=tag
        )
        n_submitted = len(results.submitted)

        return n_submitted

    def get_zpe(self, scale_factor: float = 1.0) -> Dict[str, Dict[str, float]]:
        """Get the zero point energy contributions to all molecules

        Args:
            scale_factor: Frequency scaling factor
        Returns:
            ZPE for each molecule
        """

        # Get all of the hessians and molecules
        records = self.get_complete_records()
        mols = self.get_geometries(records)

        # Iterate over records
        output = {}
        for label, record in records.items():
            # Get the name of the solvent
            inchi, state = label.rsplit("_", 1)

            # Get the molecule
            mol = mols[inchi][state]

            # Compute the ZPE
            zpe = compute_zpe(record.return_result, mol, scaling=scale_factor)

            # Store it
            if inchi not in output:
                output[inchi] = {}
            output[inchi][state] = zpe

        return output


def compute_ionization_potentials(geometry_data: GeometryDataset,
                                  solvation_data: SolvationEnergyDataset,
                                  hessian_data: Optional[HessianDataset] = None) -> pd.DataFrame:
    """Compute the ionization potential in different solvents

    Args:
        geometry_data: Geometry dataset
        solvation_data: Solvation energy computation dataset
        hessian_data: Hessian dataset, if desired
    Returns:
        Dataframe of all of the computation information
    """

    # Get the energies in vacuum
    vacuum_energies = geometry_data.get_energies()
    vacuum_energies = dict((i, g) for i, g in vacuum_energies.items() if len(g) == 3)
    logger.info(f'Found {len(vacuum_energies)} calculations with all the geometries')

    # Get the ZPEs
    zpes = None
    if hessian_data is not None:
        zpes = hessian_data.get_zpe()
        logging.info(f'Retrieved ZPEs for {len(zpes)} molecules')

        # Downselect for only the molecules with ZPEs for all three geometries
        vacuum_energies = dict((i, g) for i, g in vacuum_energies.items() if len(zpes.get(i, {})) == 3)

    # Get the entries in each solvent
    solvent_energies = solvation_data.get_energies()
    logger.info(f'Found {len(vacuum_energies)} calculations with at least one solvent')

    # Compute the IP and EA for each molecule in each solvent
    results = []
    for inchi in vacuum_energies.keys():
        # Get the vacuum and energies in solvents
        vac_eng = vacuum_energies[inchi]
        all_solv_eng = solvent_energies.get(inchi)

        # Compute the energy of the neutral
        neutral_vac_eng = vac_eng['neutral']
        if zpes is not None:
            neutral_vac_eng += zpes[inchi]['neutral']

        # Initialize the output record
        mol = Chem.MolFromInchi(inchi)
        data = {'inchi': inchi, 'smiles': Chem.MolToSmiles(mol),
                'inchi_key': Chem.MolToInchiKey(mol)}

        # Compute the EA and IP in each solvent we have
        for label, name in zip(['reduced', 'oxidized'], ['EA', 'IP']):
            # Prefactor
            p = -1 if name == "EA" else 1

            # Compute the energy of the charged
            charged_vac_eng = vac_eng[label]
            if zpes is not None:
                charged_vac_eng += zpes[inchi][label]

            # Compute the potential in gas
            g_chg = charged_vac_eng - neutral_vac_eng
            g_chg_u = constants.ureg.Quantity(g_chg * constants.hartree2kcalmol, 'kcal/mol')
            data[name] = (p * g_chg_u / f).to("V").magnitude

            # Correct for solvent
            if all_solv_eng is None or label not in all_solv_eng:
                continue
            for solv, solv_eng in all_solv_eng[label].items():
                if solv not in all_solv_eng.get('neutral', {}):
                    continue  # We need the solvation energy for neutral and ion
                solv_neu = all_solv_eng['neutral'][solv] - vac_eng['neutral']
                solv_chg = solv_eng - vac_eng[label]
                g_solv = constants.ureg.Quantity(
                    (g_chg + solv_chg - solv_neu) * constants.hartree2kcalmol, 'kcal/mol')
                e_red = p * g_solv / (1 * f)
                data[f'{name}_{solv}'] = e_red.to("V").magnitude
        results.append(data)
    return pd.DataFrame(results)
