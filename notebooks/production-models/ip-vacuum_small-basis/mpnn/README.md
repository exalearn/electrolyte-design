# MPNN for Ionization Potential

MPNN model to go from SMILES to IP directly

## Running the Model

The `run_test.py` script trains a machine learning model and saves the results to `runs`.
The output folder changes depending on the settings, which are defined as command-line arguments
to the Python scipy.
Settings include definitions of the network architecture (e.g., number of message-passing, dense layers),
how the model is trained (e.g., number of epochs)
and randomization seed(s).
