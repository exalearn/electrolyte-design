"""Utilities for using models based on SchNet"""
from typing import Union, List, Optional
from tempfile import TemporaryDirectory
from io import BytesIO, StringIO
import os

from schnetpack.data import AtomsData, AtomsLoader
from schnetpack.atomistic import Atomwise
from schnetpack import Properties
from ase.io.xyz import read_xyz
from torch.autograd import grad
from torch import nn
import numpy as np
import torch


class TorchMessage:
    """Send a PyTorch object via pickle, enable loading on to target hardware"""

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model: Model to be sent
        """
        self.model = model
        self._pickle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Save the model with pickle
        model_pkl = BytesIO()
        torch.save(self.model, model_pkl)

        # Store it
        state['model'] = None
        state['_pickle'] = model_pkl.getvalue()
        return state

    def get_model(self, map_location: Union[str, torch.device] = 'cpu'):
        """Load the cached model into memory

        Args:
            map_location: Where to copy the device
        Returns:
            Deserialized model, moved to the target resource
        """
        if self.model is None:
            self.model = torch.load(BytesIO(self._pickle), map_location=map_location)
            self._pickle = None
        else:
            self.model.to(map_location)
        return self.model


class Moleculewise(Atomwise):

    def forward(self, inputs):
        r"""
        predicts atomwise property
        """
        atom_mask = inputs[Properties.atom_mask]

        # Pool over atoms
        inputs['representation'] = self.atom_pool(inputs['representation'], atom_mask)

        # run prediction
        y = self.out_net(inputs)
        y = self.standardize(y)

        # collect results
        result = {self.property: y}

        if self.derivative:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                result[self.property],
                inputs[Properties.R],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=self.create_graph,
                retain_graph=True,
            )[0]
            result[self.derivative] = sign * dy
        return result


class AddToRepresentation(nn.Module):
    """Add features from another source to atomic representations

    These additional features could be both for individual atoms or for the
    entire molecule. For the second case, this class will duplicate the property for each atom
    before concatenating the value with the representation.

    Properties are often taken from the input dictionary

    Args:
        additional_props ([string]): List of other properties to use as inputs
    """

    def __init__(self, additional_props: List[str]):
        super(AddToRepresentation, self).__init__()
        self.additional_props = additional_props.copy()

    def forward(self, inputs):
        # Get the representation
        rep = inputs['representation']
        n_atoms = rep.shape[1]  # Use for expanding properties

        # Append the additional props
        output = [rep]
        for p in self.additional_props:
            x = inputs[p]
            if x.dim() == 1:  # Per-molecule properties
                x = torch.unsqueeze(torch.unsqueeze(x, -1).expand(-1, n_atoms), -1)
            elif x.dim() == 2:
                x = torch.unsqueeze(x, -1)
            output.append(x)
        return torch.cat(output, -1)


def evaluate_schnet(models: List[Union[TorchMessage, torch.nn.Module]],
                    molecules: List[str], property_name: str,
                    batch_size: int = 64, device: str = 'cpu',
                    base_property: Optional[List[float]] = None) -> np.ndarray:
    """Run inference for a machine learning model

    Args:
        models: List of models to evaluate. Either a SchNet model or
           the bytes corresponding to a serialized model
        molecules: XYZ-format structures of molecules to be evaluate
        property_name: Name of the property being predicted
        batch_size: Number of molecules to evaluate per batch
        device: Device on which to run the computation
        base_property: If using a delta-learning model, the value
            of the property at a lower fidelity
    """

    # Make sure the models are converted to Torch models
    if isinstance(models[0], TorchMessage):
        models = [m.get_model(device) for m in models]

    # Make the dataset
    with TemporaryDirectory() as td:
        # Convert the molecules to ase.Atoms objects
        atoms = [next(read_xyz(StringIO(x), slice(None))) for x in molecules]

        # Save the data to an ASE Atoms database
        run_file = os.path.join(td, 'run_data.db')
        db = AtomsData(run_file, available_properties=[])
        db.add_systems(atoms, [{} for _ in atoms])

        # Build the data loader
        loader = AtomsLoader(db, batch_size=batch_size)

        # Run the models
        y_preds = []
        for model in models:
            y_pred = []
            model.to(device)  # Move the model to the device
            for batch in loader:
                # Push the batch to the device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Run it and save results
                pred = model(batch)
                y_pred.append(pred[property_name].detach().cpu().numpy())
            y_preds.append(np.squeeze(np.concatenate(y_pred)))

        # If needed, add in the base property
        if base_property is not None:
            for y_pred in y_preds:
                y_pred += base_property
        return np.vstack(y_preds).T
